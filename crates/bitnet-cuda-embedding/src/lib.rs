//! CUDA embedding kernel types for `BitNet` LLM inference.
//!
//! This crate provides the data structures and logic for token and position
//! embedding lookup, batch embedding dispatch, embedding normalization, and
//! gradient accumulation types used as the foundation for CUDA-accelerated
//! embedding kernels.
//!
//! # Overview
//!
//! ```
//! use bitnet_cuda_embedding::*;
//!
//! // Build an embedding table and look up token vectors.
//! let table = EmbeddingTable::new(8, 4); // vocab=8, dim=4
//! let vec = table.lookup(2);
//! assert_eq!(vec.len(), 4);
//! ```

use bitnet_common::types::Device;

// ---------------------------------------------------------------------------
// Token embedding lookup
// ---------------------------------------------------------------------------

/// A contiguous embedding weight table stored in row-major order.
///
/// Each row `i` holds the dense vector for vocabulary token `i`.
/// Dimensions: `[vocab_size, embedding_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingTable {
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embedding_dim: usize,
    /// Flat weight storage: `vocab_size * embedding_dim` elements.
    pub weights: Vec<f32>,
}

impl EmbeddingTable {
    /// Create a zero-initialised embedding table.
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self { vocab_size, embedding_dim, weights: vec![0.0; vocab_size * embedding_dim] }
    }

    /// Create a table from pre-existing weights.
    ///
    /// # Panics
    ///
    /// Panics when `weights.len() != vocab_size * embedding_dim`.
    pub fn from_weights(vocab_size: usize, embedding_dim: usize, weights: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            vocab_size * embedding_dim,
            "weights length mismatch: expected {}, got {}",
            vocab_size * embedding_dim,
            weights.len(),
        );
        Self { vocab_size, embedding_dim, weights }
    }

    /// Return the embedding vector for `token_id`.
    ///
    /// # Panics
    ///
    /// Panics when `token_id >= vocab_size`.
    pub fn lookup(&self, token_id: u32) -> &[f32] {
        let id = token_id as usize;
        assert!(
            id < self.vocab_size,
            "token_id {id} out of range (vocab_size={})",
            self.vocab_size
        );
        let start = id * self.embedding_dim;
        &self.weights[start..start + self.embedding_dim]
    }

    /// Mutable access to the embedding vector for `token_id`.
    ///
    /// # Panics
    ///
    /// Panics when `token_id >= vocab_size`.
    pub fn lookup_mut(&mut self, token_id: u32) -> &mut [f32] {
        let id = token_id as usize;
        assert!(
            id < self.vocab_size,
            "token_id {id} out of range (vocab_size={})",
            self.vocab_size
        );
        let start = id * self.embedding_dim;
        &mut self.weights[start..start + self.embedding_dim]
    }

    /// Total number of scalar parameters in the table.
    pub const fn num_parameters(&self) -> usize {
        self.vocab_size * self.embedding_dim
    }
}

// ---------------------------------------------------------------------------
// Position embedding
// ---------------------------------------------------------------------------

/// Strategy used to inject positional information into embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PositionEmbeddingKind {
    /// Learned absolute position embeddings (a lookup table indexed by position).
    Absolute,
    /// Rotary position embedding (`RoPE`) applied after token embedding.
    Rotary,
    /// Sinusoidal fixed position encoding (Transformer-original).
    Sinusoidal,
    /// `ALiBi` -- Attention with Linear Biases (no explicit position vectors).
    Alibi,
    /// No position embedding applied.
    None,
}

/// Configuration for position embedding generation.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionEmbeddingConfig {
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Embedding dimensionality (must match token embedding dim).
    pub embedding_dim: usize,
    /// Kind of position embedding.
    pub kind: PositionEmbeddingKind,
    /// Base frequency for `RoPE` / sinusoidal (ignored for Absolute / `ALiBi`).
    pub base_freq: f32,
}

impl PositionEmbeddingConfig {
    /// Create a new config with sane defaults (Rotary, base 10000).
    pub const fn new(max_seq_len: usize, embedding_dim: usize) -> Self {
        Self {
            max_seq_len,
            embedding_dim,
            kind: PositionEmbeddingKind::Rotary,
            base_freq: 10_000.0,
        }
    }

    /// Builder helper -- set the kind.
    #[must_use]
    pub const fn with_kind(mut self, kind: PositionEmbeddingKind) -> Self {
        self.kind = kind;
        self
    }

    /// Builder helper -- set the base frequency.
    #[must_use]
    pub const fn with_base_freq(mut self, freq: f32) -> Self {
        self.base_freq = freq;
        self
    }
}

/// Absolute position embedding table.
///
/// Dimensions: `[max_seq_len, embedding_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionEmbeddingTable {
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Dimensionality.
    pub embedding_dim: usize,
    /// Flat weight storage.
    pub weights: Vec<f32>,
}

impl PositionEmbeddingTable {
    /// Create a zero-initialised position embedding table.
    pub fn new(max_seq_len: usize, embedding_dim: usize) -> Self {
        Self { max_seq_len, embedding_dim, weights: vec![0.0; max_seq_len * embedding_dim] }
    }

    /// Lookup the position vector for a given position index.
    ///
    /// # Panics
    ///
    /// Panics when `position >= max_seq_len`.
    pub fn lookup(&self, position: usize) -> &[f32] {
        assert!(
            position < self.max_seq_len,
            "position {position} out of range (max_seq_len={})",
            self.max_seq_len,
        );
        let start = position * self.embedding_dim;
        &self.weights[start..start + self.embedding_dim]
    }

    /// Total parameter count.
    pub const fn num_parameters(&self) -> usize {
        self.max_seq_len * self.embedding_dim
    }
}

/// Generate sinusoidal position encodings.
///
/// Returns a flat `Vec<f32>` of shape `[max_seq_len, embedding_dim]`.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn generate_sinusoidal(max_seq_len: usize, embedding_dim: usize, base_freq: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; max_seq_len * embedding_dim];
    let dim_f = embedding_dim as f64;
    for pos in 0..max_seq_len {
        for i in 0..embedding_dim {
            let exponent = (2 * (i / 2)) as f64 / dim_f;
            let angle = (pos as f64) / f64::from(base_freq).powf(exponent);
            out[pos * embedding_dim + i] =
                if i % 2 == 0 { angle.sin() as f32 } else { angle.cos() as f32 };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Batch embedding
// ---------------------------------------------------------------------------

/// A batch of token-id sequences to embed in a single kernel launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEmbeddingRequest {
    /// Token IDs per sequence, stored contiguously.
    pub token_ids: Vec<u32>,
    /// Number of tokens in each sequence (sum must equal `token_ids.len()`).
    pub seq_lengths: Vec<usize>,
    /// Target device for the output embeddings.
    pub device: Device,
}

impl BatchEmbeddingRequest {
    /// Create a new batch request.
    ///
    /// # Panics
    ///
    /// Panics when the sum of `seq_lengths` does not match `token_ids.len()`.
    pub fn new(token_ids: Vec<u32>, seq_lengths: Vec<usize>, device: Device) -> Self {
        let total: usize = seq_lengths.iter().sum();
        assert_eq!(
            total,
            token_ids.len(),
            "seq_lengths sum ({total}) != token_ids.len() ({})",
            token_ids.len(),
        );
        Self { token_ids, seq_lengths, device }
    }

    /// Number of sequences in the batch.
    pub const fn batch_size(&self) -> usize {
        self.seq_lengths.len()
    }

    /// Total number of tokens across all sequences.
    pub const fn total_tokens(&self) -> usize {
        self.token_ids.len()
    }

    /// Return the token IDs for a specific sequence index.
    ///
    /// # Panics
    ///
    /// Panics when `seq_idx >= batch_size()`.
    pub fn sequence_tokens(&self, seq_idx: usize) -> &[u32] {
        assert!(seq_idx < self.seq_lengths.len(), "seq_idx out of range");
        let start: usize = self.seq_lengths[..seq_idx].iter().sum();
        let len = self.seq_lengths[seq_idx];
        &self.token_ids[start..start + len]
    }
}

/// Output of a batched embedding lookup.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchEmbeddingOutput {
    /// Flat embedding data: `[total_tokens, embedding_dim]`.
    pub embeddings: Vec<f32>,
    /// Embedding dimensionality.
    pub embedding_dim: usize,
    /// Per-sequence lengths (mirrors the request).
    pub seq_lengths: Vec<usize>,
}

impl BatchEmbeddingOutput {
    /// Return the embedded vectors for a specific sequence.
    ///
    /// # Panics
    ///
    /// Panics when `seq_idx >= seq_lengths.len()`.
    pub fn sequence_embeddings(&self, seq_idx: usize) -> &[f32] {
        assert!(seq_idx < self.seq_lengths.len(), "seq_idx out of range");
        let token_offset: usize = self.seq_lengths[..seq_idx].iter().sum();
        let n_tokens = self.seq_lengths[seq_idx];
        let start = token_offset * self.embedding_dim;
        let end = start + n_tokens * self.embedding_dim;
        &self.embeddings[start..end]
    }

    /// Total number of tokens across all sequences.
    pub fn total_tokens(&self) -> usize {
        self.seq_lengths.iter().sum()
    }
}

/// Execute a batched embedding lookup on the given table.
///
/// This is the CPU reference path; a CUDA kernel would replace the inner
/// loop with a GPU launch.
pub fn batch_embed(
    table: &EmbeddingTable,
    request: &BatchEmbeddingRequest,
) -> BatchEmbeddingOutput {
    let total: usize = request.total_tokens();
    let dim = table.embedding_dim;
    let mut embeddings = Vec::with_capacity(total * dim);
    for &tid in &request.token_ids {
        embeddings.extend_from_slice(table.lookup(tid));
    }
    BatchEmbeddingOutput {
        embeddings,
        embedding_dim: dim,
        seq_lengths: request.seq_lengths.clone(),
    }
}

// ---------------------------------------------------------------------------
// Embedding normalization
// ---------------------------------------------------------------------------

/// Normalization strategy applied to embedding vectors after lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingNormKind {
    /// No normalization.
    None,
    /// L2 (unit-length) normalization.
    L2,
    /// Layer normalization (zero mean, unit variance per vector).
    LayerNorm,
    /// RMS normalization.
    RmsNorm,
}

/// Configuration for embedding normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingNormConfig {
    /// Kind of normalization.
    pub kind: EmbeddingNormKind,
    /// Small epsilon to avoid division by zero.
    pub eps: f32,
}

impl Default for EmbeddingNormConfig {
    fn default() -> Self {
        Self { kind: EmbeddingNormKind::None, eps: 1e-5 }
    }
}

impl EmbeddingNormConfig {
    /// Builder helper -- set the normalization kind.
    #[must_use]
    pub const fn with_kind(mut self, kind: EmbeddingNormKind) -> Self {
        self.kind = kind;
        self
    }
}

/// Apply normalization in-place to a single embedding vector.
pub fn normalize_embedding(vec: &mut [f32], config: &EmbeddingNormConfig) {
    match config.kind {
        EmbeddingNormKind::None => {}
        EmbeddingNormKind::L2 => normalize_l2(vec, config.eps),
        EmbeddingNormKind::LayerNorm => normalize_layer_norm(vec, config.eps),
        EmbeddingNormKind::RmsNorm => normalize_rms(vec, config.eps),
    }
}

/// L2 normalization: `v / ||v||_2`.
fn normalize_l2(vec: &mut [f32], eps: f32) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(eps);
    for x in vec.iter_mut() {
        *x /= norm;
    }
}

/// Layer normalization: zero mean, unit variance.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn normalize_layer_norm(vec: &mut [f32], eps: f32) {
    let n = vec.len() as f64;
    if n == 0.0 {
        return;
    }
    let mean = vec.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = vec
        .iter()
        .map(|&x| {
            let d = f64::from(x) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = (var + f64::from(eps)).sqrt();
    for x in vec.iter_mut() {
        *x = ((f64::from(*x) - mean) / std) as f32;
    }
}

/// RMS normalization: scale by root-mean-square.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn normalize_rms(vec: &mut [f32], eps: f32) {
    let n = vec.len() as f64;
    if n == 0.0 {
        return;
    }
    let rms =
        (vec.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / n + f64::from(eps)).sqrt();
    for x in vec.iter_mut() {
        *x = (f64::from(*x) / rms) as f32;
    }
}

/// Apply normalization to every token vector in a flat embedding buffer.
///
/// `embeddings` has shape `[n_tokens, embedding_dim]`.
pub fn normalize_batch(embeddings: &mut [f32], embedding_dim: usize, config: &EmbeddingNormConfig) {
    assert!(
        embedding_dim > 0 && embeddings.len().is_multiple_of(embedding_dim),
        "embedding buffer length must be a multiple of embedding_dim",
    );
    for chunk in embeddings.chunks_mut(embedding_dim) {
        normalize_embedding(chunk, config);
    }
}

// ---------------------------------------------------------------------------
// Gradient accumulation (training-prep types)
// ---------------------------------------------------------------------------

/// Accumulated embedding gradients for sparse updates.
///
/// During back-propagation only the rows corresponding to tokens that
/// appeared in the forward pass receive non-zero gradients. This struct
/// collects those sparse updates before they are applied to the table.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingGradAccumulator {
    /// Embedding dimensionality.
    pub embedding_dim: usize,
    /// Per-token gradient sums: `(token_id, grad_vector)`.
    entries: Vec<(u32, Vec<f32>)>,
}

impl EmbeddingGradAccumulator {
    /// Create a new empty accumulator.
    pub const fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim, entries: Vec::new() }
    }

    /// Accumulate a gradient for a specific token.
    ///
    /// # Panics
    ///
    /// Panics when `grad.len() != embedding_dim`.
    pub fn accumulate(&mut self, token_id: u32, grad: &[f32]) {
        assert_eq!(
            grad.len(),
            self.embedding_dim,
            "grad length mismatch: expected {}, got {}",
            self.embedding_dim,
            grad.len(),
        );
        if let Some(entry) = self.entries.iter_mut().find(|(id, _)| *id == token_id) {
            for (dst, src) in entry.1.iter_mut().zip(grad.iter()) {
                *dst += src;
            }
        } else {
            self.entries.push((token_id, grad.to_vec()));
        }
    }

    /// Number of unique tokens with accumulated gradients.
    pub const fn num_unique_tokens(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over `(token_id, gradient_vector)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &[f32])> {
        self.entries.iter().map(|(id, g)| (*id, g.as_slice()))
    }

    /// Clear all accumulated gradients.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Apply accumulated gradients to an embedding table with a learning rate.
    ///
    /// Performs: `weight[token] -= lr * grad[token]`.
    pub fn apply(&self, table: &mut EmbeddingTable, lr: f32) {
        for (token_id, grad) in &self.entries {
            let row = table.lookup_mut(*token_id);
            for (w, g) in row.iter_mut().zip(grad.iter()) {
                *w -= lr * g;
            }
        }
    }
}

/// Optimizer state for a single embedding row (Adam).
#[derive(Debug, Clone, PartialEq)]
pub struct AdamEmbeddingState {
    /// First moment estimate (mean of gradients).
    pub m: Vec<f32>,
    /// Second moment estimate (mean of squared gradients).
    pub v: Vec<f32>,
    /// Number of update steps applied.
    pub step: u64,
}

impl AdamEmbeddingState {
    /// Create a zero-initialised Adam state for a given dimensionality.
    pub fn new(embedding_dim: usize) -> Self {
        Self { m: vec![0.0; embedding_dim], v: vec![0.0; embedding_dim], step: 0 }
    }
}

// ---------------------------------------------------------------------------
// Kernel launch descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a CUDA embedding kernel launch.
///
/// Captures the parameters that a CUDA kernel would need when launched on
/// the GPU.  On CPU this is informational only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingKernelDescriptor {
    /// Target device.
    pub device: Device,
    /// Block size (threads per block) for the CUDA launch grid.
    pub block_size: u32,
    /// Whether to use shared-memory tiling for large embedding dims.
    pub use_shared_memory: bool,
    /// Stream index for concurrent kernel execution.
    pub stream_index: u32,
}

impl Default for EmbeddingKernelDescriptor {
    fn default() -> Self {
        Self { device: Device::Cpu, block_size: 256, use_shared_memory: false, stream_index: 0 }
    }
}

impl EmbeddingKernelDescriptor {
    /// Builder helper – set the target device.
    #[must_use]
    pub const fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Builder helper – set the block size.
    #[must_use]
    pub const fn with_block_size(mut self, block_size: u32) -> Self {
        self.block_size = block_size;
        self
    }

    /// Builder helper – enable shared-memory tiling.
    #[must_use]
    pub const fn with_shared_memory(mut self, enabled: bool) -> Self {
        self.use_shared_memory = enabled;
        self
    }

    /// Builder helper – set the CUDA stream index.
    #[must_use]
    pub const fn with_stream_index(mut self, idx: u32) -> Self {
        self.stream_index = idx;
        self
    }

    /// Compute the grid size for a given total number of elements.
    pub const fn grid_size(&self, total_elements: u32) -> u32 {
        total_elements.div_ceil(self.block_size)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- EmbeddingTable -----------------------------------------------------

    #[test]
    fn test_embedding_table_new_dimensions() {
        let t = EmbeddingTable::new(100, 64);
        assert_eq!(t.vocab_size, 100);
        assert_eq!(t.embedding_dim, 64);
        assert_eq!(t.weights.len(), 100 * 64);
    }

    #[test]
    fn test_embedding_table_new_zero_init() {
        let t = EmbeddingTable::new(4, 2);
        assert!(t.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_embedding_table_from_weights() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = EmbeddingTable::from_weights(3, 2, w.clone());
        assert_eq!(t.weights, w);
    }

    #[test]
    #[should_panic(expected = "weights length mismatch")]
    fn test_embedding_table_from_weights_mismatch() {
        EmbeddingTable::from_weights(3, 2, vec![1.0; 5]);
    }

    #[test]
    fn test_embedding_table_lookup() {
        let w = vec![10.0, 20.0, 30.0, 40.0];
        let t = EmbeddingTable::from_weights(2, 2, w);
        assert_eq!(t.lookup(0), &[10.0, 20.0]);
        assert_eq!(t.lookup(1), &[30.0, 40.0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_embedding_table_lookup_out_of_range() {
        let t = EmbeddingTable::new(2, 2);
        t.lookup(2);
    }

    #[test]
    fn test_embedding_table_lookup_mut() {
        let mut t = EmbeddingTable::new(2, 2);
        t.lookup_mut(1)[0] = 42.0;
        assert_eq!(t.lookup(1)[0], 42.0);
    }

    #[test]
    fn test_embedding_table_num_parameters() {
        let t = EmbeddingTable::new(50, 128);
        assert_eq!(t.num_parameters(), 50 * 128);
    }

    #[test]
    fn test_embedding_table_single_element() {
        let t = EmbeddingTable::new(1, 1);
        assert_eq!(t.lookup(0), &[0.0]);
    }

    #[test]
    fn test_embedding_table_large() {
        let t = EmbeddingTable::new(32_000, 4096);
        assert_eq!(t.num_parameters(), 32_000 * 4096);
    }

    // -- PositionEmbeddingKind / Config -------------------------------------

    #[test]
    fn test_position_kind_debug() {
        assert_eq!(format!("{:?}", PositionEmbeddingKind::Rotary), "Rotary");
    }

    #[test]
    fn test_position_config_defaults() {
        let c = PositionEmbeddingConfig::new(512, 64);
        assert_eq!(c.max_seq_len, 512);
        assert_eq!(c.embedding_dim, 64);
        assert_eq!(c.kind, PositionEmbeddingKind::Rotary);
        assert!((c.base_freq - 10_000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_position_config_with_kind() {
        let c = PositionEmbeddingConfig::new(128, 32).with_kind(PositionEmbeddingKind::Absolute);
        assert_eq!(c.kind, PositionEmbeddingKind::Absolute);
    }

    #[test]
    fn test_position_config_with_base_freq() {
        let c = PositionEmbeddingConfig::new(128, 32).with_base_freq(500_000.0);
        assert!((c.base_freq - 500_000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_position_config_chained_builders() {
        let c = PositionEmbeddingConfig::new(256, 16)
            .with_kind(PositionEmbeddingKind::Sinusoidal)
            .with_base_freq(20_000.0);
        assert_eq!(c.kind, PositionEmbeddingKind::Sinusoidal);
        assert!((c.base_freq - 20_000.0).abs() < f32::EPSILON);
    }

    // -- PositionEmbeddingTable ---------------------------------------------

    #[test]
    fn test_position_table_new() {
        let t = PositionEmbeddingTable::new(128, 64);
        assert_eq!(t.max_seq_len, 128);
        assert_eq!(t.embedding_dim, 64);
        assert_eq!(t.weights.len(), 128 * 64);
    }

    #[test]
    fn test_position_table_lookup() {
        let mut t = PositionEmbeddingTable::new(4, 2);
        t.weights[2] = 1.0; // position 1, dim 0
        assert_eq!(t.lookup(1), &[1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_position_table_lookup_out_of_range() {
        let t = PositionEmbeddingTable::new(4, 2);
        t.lookup(4);
    }

    #[test]
    fn test_position_table_num_parameters() {
        let t = PositionEmbeddingTable::new(512, 128);
        assert_eq!(t.num_parameters(), 512 * 128);
    }

    // -- generate_sinusoidal ------------------------------------------------

    #[test]
    fn test_sinusoidal_length() {
        let s = generate_sinusoidal(8, 4, 10_000.0);
        assert_eq!(s.len(), 8 * 4);
    }

    #[test]
    fn test_sinusoidal_position_zero() {
        let s = generate_sinusoidal(1, 4, 10_000.0);
        // sin(0) == 0, cos(0) == 1 for all frequencies at position 0
        assert!((s[0] - 0.0).abs() < 1e-6); // sin(0)
        assert!((s[1] - 1.0).abs() < 1e-6); // cos(0)
    }

    #[test]
    fn test_sinusoidal_not_all_zero() {
        let s = generate_sinusoidal(16, 8, 10_000.0);
        assert!(s.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn test_sinusoidal_different_base_freq() {
        let a = generate_sinusoidal(4, 4, 10_000.0);
        let b = generate_sinusoidal(4, 4, 20_000.0);
        // Position 0 is the same, but others differ.
        assert_ne!(a, b);
    }

    // -- BatchEmbeddingRequest ----------------------------------------------

    #[test]
    fn test_batch_request_creation() {
        let r = BatchEmbeddingRequest::new(vec![1, 2, 3], vec![2, 1], Device::Cpu);
        assert_eq!(r.batch_size(), 2);
        assert_eq!(r.total_tokens(), 3);
    }

    #[test]
    #[should_panic(expected = "seq_lengths sum")]
    fn test_batch_request_length_mismatch() {
        BatchEmbeddingRequest::new(vec![1, 2], vec![1, 2], Device::Cpu);
    }

    #[test]
    fn test_batch_request_sequence_tokens() {
        let r = BatchEmbeddingRequest::new(vec![10, 20, 30, 40], vec![1, 3], Device::Cpu);
        assert_eq!(r.sequence_tokens(0), &[10]);
        assert_eq!(r.sequence_tokens(1), &[20, 30, 40]);
    }

    #[test]
    #[should_panic(expected = "seq_idx out of range")]
    fn test_batch_request_sequence_tokens_oob() {
        let r = BatchEmbeddingRequest::new(vec![1], vec![1], Device::Cpu);
        r.sequence_tokens(1);
    }

    #[test]
    fn test_batch_request_single_sequence() {
        let r = BatchEmbeddingRequest::new(vec![5, 6, 7], vec![3], Device::Cpu);
        assert_eq!(r.batch_size(), 1);
        assert_eq!(r.sequence_tokens(0), &[5, 6, 7]);
    }

    #[test]
    fn test_batch_request_empty_sequence() {
        let r = BatchEmbeddingRequest::new(vec![1], vec![0, 1], Device::Cpu);
        assert_eq!(r.sequence_tokens(0), &[] as &[u32]);
        assert_eq!(r.sequence_tokens(1), &[1]);
    }

    // -- BatchEmbeddingOutput -----------------------------------------------

    #[test]
    fn test_batch_output_sequence_embeddings() {
        let out = BatchEmbeddingOutput {
            embeddings: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            embedding_dim: 2,
            seq_lengths: vec![1, 2],
        };
        assert_eq!(out.sequence_embeddings(0), &[1.0, 2.0]);
        assert_eq!(out.sequence_embeddings(1), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_batch_output_total_tokens() {
        let out = BatchEmbeddingOutput {
            embeddings: vec![0.0; 12],
            embedding_dim: 4,
            seq_lengths: vec![1, 2],
        };
        assert_eq!(out.total_tokens(), 3);
    }

    #[test]
    #[should_panic(expected = "seq_idx out of range")]
    fn test_batch_output_sequence_oob() {
        let out =
            BatchEmbeddingOutput { embeddings: vec![], embedding_dim: 2, seq_lengths: vec![] };
        out.sequence_embeddings(0);
    }

    // -- batch_embed --------------------------------------------------------

    #[test]
    fn test_batch_embed_basic() {
        let w = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let table = EmbeddingTable::from_weights(3, 2, w);
        let req = BatchEmbeddingRequest::new(vec![0, 2, 1], vec![2, 1], Device::Cpu);
        let out = batch_embed(&table, &req);
        assert_eq!(out.embedding_dim, 2);
        assert_eq!(out.embeddings, vec![10.0, 20.0, 50.0, 60.0, 30.0, 40.0]);
    }

    #[test]
    fn test_batch_embed_single_token() {
        let table = EmbeddingTable::from_weights(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let req = BatchEmbeddingRequest::new(vec![1], vec![1], Device::Cpu);
        let out = batch_embed(&table, &req);
        assert_eq!(out.embeddings, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_batch_embed_repeated_tokens() {
        let table = EmbeddingTable::from_weights(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let req = BatchEmbeddingRequest::new(vec![0, 0, 0], vec![3], Device::Cpu);
        let out = batch_embed(&table, &req);
        assert_eq!(out.embeddings, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_batch_embed_preserves_seq_lengths() {
        let table = EmbeddingTable::new(10, 4);
        let req = BatchEmbeddingRequest::new(vec![0, 1, 2, 3, 4], vec![2, 3], Device::Cpu);
        let out = batch_embed(&table, &req);
        assert_eq!(out.seq_lengths, vec![2, 3]);
    }

    // -- EmbeddingNormConfig / normalize ------------------------------------

    #[test]
    fn test_norm_config_default() {
        let c = EmbeddingNormConfig::default();
        assert_eq!(c.kind, EmbeddingNormKind::None);
        assert!((c.eps - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn test_norm_config_with_kind() {
        let c = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::L2);
        assert_eq!(c.kind, EmbeddingNormKind::L2);
    }

    #[test]
    fn test_normalize_none_noop() {
        let mut v = vec![1.0, 2.0, 3.0];
        let cfg = EmbeddingNormConfig::default();
        normalize_embedding(&mut v, &cfg);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_normalize_l2_unit_length() {
        let mut v = vec![3.0, 4.0];
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::L2);
        normalize_embedding(&mut v, &cfg);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_l2_direction() {
        let mut v = vec![3.0, 4.0];
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::L2);
        normalize_embedding(&mut v, &cfg);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_l2_all_zeros() {
        let mut v = vec![0.0, 0.0, 0.0];
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::L2);
        normalize_embedding(&mut v, &cfg);
        // Should not produce NaN – eps clamps the denominator.
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_normalize_layer_norm_mean_zero() {
        let mut v = vec![1.0, 3.0, 5.0, 7.0];
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::LayerNorm);
        normalize_embedding(&mut v, &cfg);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_normalize_layer_norm_unit_var() {
        let mut v = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::LayerNorm);
        normalize_embedding(&mut v, &cfg);
        let n = v.len() as f32;
        let mean: f32 = v.iter().sum::<f32>() / n;
        let var: f32 = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_normalize_rms() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::RmsNorm);
        normalize_embedding(&mut v, &cfg);
        let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_normalize_rms_preserves_sign() {
        let mut v = vec![-2.0, 3.0];
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::RmsNorm);
        normalize_embedding(&mut v, &cfg);
        assert!(v[0] < 0.0);
        assert!(v[1] > 0.0);
    }

    #[test]
    fn test_normalize_batch() {
        let mut buf = vec![3.0, 4.0, 0.0, 0.0, 6.0, 8.0];
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::L2);
        normalize_batch(&mut buf, 2, &cfg);
        // First vector [3,4] -> [0.6, 0.8]
        assert!((buf[0] - 0.6).abs() < 1e-5);
        assert!((buf[1] - 0.8).abs() < 1e-5);
        // Third vector [6,8] -> [0.6, 0.8]
        assert!((buf[4] - 0.6).abs() < 1e-5);
        assert!((buf[5] - 0.8).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "multiple of embedding_dim")]
    fn test_normalize_batch_bad_length() {
        let mut buf = vec![1.0, 2.0, 3.0];
        let cfg = EmbeddingNormConfig::default();
        normalize_batch(&mut buf, 2, &cfg);
    }

    // -- EmbeddingGradAccumulator -------------------------------------------

    #[test]
    fn test_grad_accum_new() {
        let a = EmbeddingGradAccumulator::new(4);
        assert_eq!(a.embedding_dim, 4);
        assert_eq!(a.num_unique_tokens(), 0);
    }

    #[test]
    fn test_grad_accum_single_token() {
        let mut a = EmbeddingGradAccumulator::new(2);
        a.accumulate(5, &[1.0, 2.0]);
        assert_eq!(a.num_unique_tokens(), 1);
        let (id, g) = a.iter().next().unwrap();
        assert_eq!(id, 5);
        assert_eq!(g, &[1.0, 2.0]);
    }

    #[test]
    fn test_grad_accum_merge_same_token() {
        let mut a = EmbeddingGradAccumulator::new(2);
        a.accumulate(3, &[1.0, 2.0]);
        a.accumulate(3, &[0.5, 0.5]);
        assert_eq!(a.num_unique_tokens(), 1);
        let (_, g) = a.iter().next().unwrap();
        assert!((g[0] - 1.5).abs() < 1e-6);
        assert!((g[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_grad_accum_different_tokens() {
        let mut a = EmbeddingGradAccumulator::new(2);
        a.accumulate(0, &[1.0, 0.0]);
        a.accumulate(1, &[0.0, 1.0]);
        assert_eq!(a.num_unique_tokens(), 2);
    }

    #[test]
    #[should_panic(expected = "grad length mismatch")]
    fn test_grad_accum_wrong_dim() {
        let mut a = EmbeddingGradAccumulator::new(3);
        a.accumulate(0, &[1.0, 2.0]);
    }

    #[test]
    fn test_grad_accum_clear() {
        let mut a = EmbeddingGradAccumulator::new(2);
        a.accumulate(0, &[1.0, 2.0]);
        a.clear();
        assert_eq!(a.num_unique_tokens(), 0);
    }

    #[test]
    fn test_grad_accum_apply() {
        let mut table = EmbeddingTable::from_weights(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let mut a = EmbeddingGradAccumulator::new(2);
        a.accumulate(0, &[1.0, 2.0]);
        a.apply(&mut table, 0.1);
        // weight -= lr * grad => [10 - 0.1, 20 - 0.2] = [9.9, 19.8]
        assert!((table.lookup(0)[0] - 9.9).abs() < 1e-5);
        assert!((table.lookup(0)[1] - 19.8).abs() < 1e-5);
        // Token 1 unchanged
        assert_eq!(table.lookup(1), &[30.0, 40.0]);
    }

    #[test]
    fn test_grad_accum_apply_zero_lr() {
        let mut table = EmbeddingTable::from_weights(1, 2, vec![5.0, 6.0]);
        let mut a = EmbeddingGradAccumulator::new(2);
        a.accumulate(0, &[100.0, 200.0]);
        a.apply(&mut table, 0.0);
        assert_eq!(table.lookup(0), &[5.0, 6.0]);
    }

    // -- AdamEmbeddingState -------------------------------------------------

    #[test]
    fn test_adam_state_new() {
        let s = AdamEmbeddingState::new(4);
        assert_eq!(s.m.len(), 4);
        assert_eq!(s.v.len(), 4);
        assert_eq!(s.step, 0);
        assert!(s.m.iter().all(|&x| x == 0.0));
        assert!(s.v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_adam_state_clone() {
        let s = AdamEmbeddingState::new(3);
        let s2 = s.clone();
        assert_eq!(s, s2);
    }

    // -- EmbeddingKernelDescriptor ------------------------------------------

    #[test]
    fn test_kernel_desc_default() {
        let d = EmbeddingKernelDescriptor::default();
        assert_eq!(d.device, Device::Cpu);
        assert_eq!(d.block_size, 256);
        assert!(!d.use_shared_memory);
        assert_eq!(d.stream_index, 0);
    }

    #[test]
    fn test_kernel_desc_with_device() {
        let d = EmbeddingKernelDescriptor::default().with_device(Device::Cuda(0));
        assert_eq!(d.device, Device::Cuda(0));
    }

    #[test]
    fn test_kernel_desc_with_block_size() {
        let d = EmbeddingKernelDescriptor::default().with_block_size(512);
        assert_eq!(d.block_size, 512);
    }

    #[test]
    fn test_kernel_desc_with_shared_memory() {
        let d = EmbeddingKernelDescriptor::default().with_shared_memory(true);
        assert!(d.use_shared_memory);
    }

    #[test]
    fn test_kernel_desc_with_stream_index() {
        let d = EmbeddingKernelDescriptor::default().with_stream_index(3);
        assert_eq!(d.stream_index, 3);
    }

    #[test]
    fn test_kernel_desc_grid_size_exact() {
        let d = EmbeddingKernelDescriptor::default().with_block_size(256);
        assert_eq!(d.grid_size(256), 1);
        assert_eq!(d.grid_size(512), 2);
    }

    #[test]
    fn test_kernel_desc_grid_size_round_up() {
        let d = EmbeddingKernelDescriptor::default().with_block_size(256);
        assert_eq!(d.grid_size(257), 2);
        assert_eq!(d.grid_size(1), 1);
    }

    #[test]
    fn test_kernel_desc_chained_builders() {
        let d = EmbeddingKernelDescriptor::default()
            .with_device(Device::Cuda(1))
            .with_block_size(128)
            .with_shared_memory(true)
            .with_stream_index(2);
        assert_eq!(d.device, Device::Cuda(1));
        assert_eq!(d.block_size, 128);
        assert!(d.use_shared_memory);
        assert_eq!(d.stream_index, 2);
    }

    // -- Integration / cross-cutting ----------------------------------------

    #[test]
    fn test_embed_then_normalize() {
        let table = EmbeddingTable::from_weights(2, 2, vec![3.0, 4.0, 6.0, 8.0]);
        let req = BatchEmbeddingRequest::new(vec![0, 1], vec![1, 1], Device::Cpu);
        let mut out = batch_embed(&table, &req);
        let cfg = EmbeddingNormConfig::default().with_kind(EmbeddingNormKind::L2);
        normalize_batch(&mut out.embeddings, out.embedding_dim, &cfg);
        // [3,4] -> [0.6, 0.8], [6,8] -> [0.6, 0.8]
        assert!((out.embeddings[0] - 0.6).abs() < 1e-5);
        assert!((out.embeddings[1] - 0.8).abs() < 1e-5);
        assert!((out.embeddings[2] - 0.6).abs() < 1e-5);
        assert!((out.embeddings[3] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_embed_then_grad_update() {
        let mut table = EmbeddingTable::from_weights(3, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let req = BatchEmbeddingRequest::new(vec![0, 2], vec![2], Device::Cpu);
        let _out = batch_embed(&table, &req);
        let mut accum = EmbeddingGradAccumulator::new(2);
        accum.accumulate(0, &[0.1, 0.1]);
        accum.accumulate(2, &[0.2, 0.2]);
        accum.apply(&mut table, 1.0);
        assert!((table.lookup(0)[0] - 0.9).abs() < 1e-5);
        assert!((table.lookup(2)[0] - 2.8).abs() < 1e-5);
        // Token 1 unchanged
        assert_eq!(table.lookup(1), &[2.0, 2.0]);
    }

    #[test]
    fn test_position_kind_equality() {
        assert_eq!(PositionEmbeddingKind::Alibi, PositionEmbeddingKind::Alibi);
        assert_ne!(PositionEmbeddingKind::Rotary, PositionEmbeddingKind::Absolute);
    }

    #[test]
    fn test_norm_kind_equality() {
        assert_eq!(EmbeddingNormKind::L2, EmbeddingNormKind::L2);
        assert_ne!(EmbeddingNormKind::RmsNorm, EmbeddingNormKind::LayerNorm);
    }

    #[test]
    fn test_batch_embed_empty_batch() {
        let table = EmbeddingTable::new(10, 4);
        let req = BatchEmbeddingRequest::new(vec![], vec![], Device::Cpu);
        let out = batch_embed(&table, &req);
        assert!(out.embeddings.is_empty());
        assert_eq!(out.total_tokens(), 0);
    }

    #[test]
    fn test_sinusoidal_deterministic() {
        let a = generate_sinusoidal(8, 4, 10_000.0);
        let b = generate_sinusoidal(8, 4, 10_000.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_grad_accum_iter_order() {
        let mut a = EmbeddingGradAccumulator::new(1);
        a.accumulate(10, &[1.0]);
        a.accumulate(5, &[2.0]);
        a.accumulate(20, &[3.0]);
        let ids: Vec<u32> = a.iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![10, 5, 20]);
    }

    #[test]
    fn test_position_embedding_none_variant() {
        let c = PositionEmbeddingConfig::new(64, 8).with_kind(PositionEmbeddingKind::None);
        assert_eq!(c.kind, PositionEmbeddingKind::None);
    }

    #[test]
    fn test_kernel_desc_grid_size_zero_elements() {
        let d = EmbeddingKernelDescriptor::default();
        assert_eq!(d.grid_size(0), 0);
    }
}
