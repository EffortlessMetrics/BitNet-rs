//! OpenCL embedding lookup and output projection (lm_head) implementations.
//!
//! Provides CPU reference implementations and OpenCL kernel sources for:
//!
//! - **Embedding lookup**: token ID → dense vector from a learned table
//! - **Output projection** (lm_head): hidden state → logits via `hidden @ weight^T`
//! - **Tied weights**: shared weight matrix for embedding and output projection
//! - **Position embeddings**: absolute position encoding (additive)
//! - **Embedding normalization**: optional RMS normalization after lookup
//!
//! The OpenCL kernel source is embedded at compile time via `include_str!`
//! from `gpu/kernels/embedding.cl`.

use bitnet_common::{KernelError, Result};

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for embedding and projection operations.
pub const EMBEDDING_CL: &str = include_str!("gpu/kernels/embedding.cl");

// ── Configuration ────────────────────────────────────────────────

/// Configuration for embedding operations.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embedding_dim: usize,
    /// Optional padding index: tokens with this ID produce zero vectors.
    pub padding_idx: Option<u32>,
}

impl EmbeddingConfig {
    /// Create a new embedding configuration.
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self { vocab_size, embedding_dim, padding_idx: None }
    }

    /// Set padding index.
    #[must_use]
    pub fn with_padding_idx(mut self, idx: u32) -> Self {
        self.padding_idx = Some(idx);
        self
    }
}

// ── EmbeddingTable ───────────────────────────────────────────────

/// Token embedding table: maps token IDs to dense vectors.
///
/// Stores a weight matrix `[vocab_size, embedding_dim]` and performs
/// lookup for a batch of token IDs.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    /// Weight matrix in row-major layout: `[vocab_size, embedding_dim]`.
    pub weight: Vec<f32>,
    /// Configuration.
    pub config: EmbeddingConfig,
}

impl EmbeddingTable {
    /// Create a new embedding table with the given weights.
    ///
    /// # Errors
    /// Returns an error if `weight.len() != vocab_size * embedding_dim`.
    pub fn new(weight: Vec<f32>, config: EmbeddingConfig) -> Result<Self> {
        let expected = config.vocab_size * config.embedding_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "embedding weight length {} != vocab_size({}) * embedding_dim({})",
                    weight.len(),
                    config.vocab_size,
                    config.embedding_dim,
                ),
            }
            .into());
        }
        Ok(Self { weight, config })
    }

    /// Look up embeddings for a batch of token IDs.
    pub fn lookup(&self, token_ids: &[u32], output: &mut [f32]) -> Result<()> {
        embedding_lookup_ref(
            token_ids,
            &self.weight,
            output,
            self.config.vocab_size,
            self.config.embedding_dim,
            self.config.padding_idx,
        )
    }
}

// ── OutputProjection ─────────────────────────────────────────────

/// Output projection layer (lm_head): hidden → logits.
///
/// Computes `logits = hidden @ weight^T` where weight is
/// `[vocab_size, hidden_size]`.
#[derive(Debug, Clone)]
pub struct OutputProjection {
    /// Weight matrix: `[vocab_size, hidden_size]`.
    pub weight: Vec<f32>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden size (must equal embedding_dim for tied weights).
    pub hidden_size: usize,
}

impl OutputProjection {
    /// Create a new output projection layer.
    pub fn new(weight: Vec<f32>, vocab_size: usize, hidden_size: usize) -> Result<Self> {
        let expected = vocab_size * hidden_size;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "projection weight length {} != vocab_size({}) * hidden_size({})",
                    weight.len(),
                    vocab_size,
                    hidden_size,
                ),
            }
            .into());
        }
        Ok(Self { weight, vocab_size, hidden_size })
    }

    /// Project hidden states to logits.
    pub fn forward(&self, hidden: &[f32], output: &mut [f32], seq_len: usize) -> Result<()> {
        output_projection_ref(
            hidden,
            &self.weight,
            output,
            seq_len,
            self.hidden_size,
            self.vocab_size,
        )
    }
}

// ── TiedEmbedding ────────────────────────────────────────────────

/// Tied embedding: shares the same weight for lookup and output projection.
///
/// Many LLMs share the embedding weight matrix with the final lm_head
/// projection to reduce parameter count.
#[derive(Debug, Clone)]
pub struct TiedEmbedding {
    /// Shared weight matrix: `[vocab_size, embedding_dim]`.
    pub weight: Vec<f32>,
    /// Configuration for the embedding.
    pub config: EmbeddingConfig,
}

impl TiedEmbedding {
    /// Create a tied embedding with shared weights.
    pub fn new(weight: Vec<f32>, config: EmbeddingConfig) -> Result<Self> {
        let expected = config.vocab_size * config.embedding_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tied weight length {} != vocab_size({}) * embedding_dim({})",
                    weight.len(),
                    config.vocab_size,
                    config.embedding_dim,
                ),
            }
            .into());
        }
        Ok(Self { weight, config })
    }

    /// Look up embeddings (forward direction).
    pub fn lookup(&self, token_ids: &[u32], output: &mut [f32]) -> Result<()> {
        embedding_lookup_ref(
            token_ids,
            &self.weight,
            output,
            self.config.vocab_size,
            self.config.embedding_dim,
            self.config.padding_idx,
        )
    }

    /// Project hidden states to logits (reverse direction, lm_head).
    pub fn project(
        &self,
        hidden: &[f32],
        output: &mut [f32],
        seq_len: usize,
    ) -> Result<()> {
        output_projection_ref(
            hidden,
            &self.weight,
            output,
            seq_len,
            self.config.embedding_dim,
            self.config.vocab_size,
        )
    }

    /// Get a reference to the shared weight.
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }
}

// ── PositionEmbedding ────────────────────────────────────────────

/// Absolute position embedding table.
///
/// Stores learned position vectors `[max_seq_len, embedding_dim]` and
/// adds them element-wise to token embeddings.
#[derive(Debug, Clone)]
pub struct PositionEmbedding {
    /// Position weight matrix: `[max_seq_len, embedding_dim]`.
    pub weight: Vec<f32>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Embedding dimension.
    pub embedding_dim: usize,
}

impl PositionEmbedding {
    /// Create a new position embedding table.
    pub fn new(weight: Vec<f32>, max_seq_len: usize, embedding_dim: usize) -> Result<Self> {
        let expected = max_seq_len * embedding_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "position weight length {} != max_seq_len({}) * embedding_dim({})",
                    weight.len(),
                    max_seq_len,
                    embedding_dim,
                ),
            }
            .into());
        }
        Ok(Self { weight, max_seq_len, embedding_dim })
    }

    /// Add position embeddings to token embeddings in-place.
    ///
    /// `embeddings` has shape `[seq_len, embedding_dim]`.
    /// Position offset is the starting position index.
    pub fn add_to(
        &self,
        embeddings: &mut [f32],
        seq_len: usize,
        position_offset: usize,
    ) -> Result<()> {
        let d = self.embedding_dim;
        if position_offset + seq_len > self.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "position_offset({}) + seq_len({}) exceeds max_seq_len({})",
                    position_offset, seq_len, self.max_seq_len,
                ),
            }
            .into());
        }
        if embeddings.len() < seq_len * d {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "embeddings length {} < seq_len({}) * embedding_dim({})",
                    embeddings.len(),
                    seq_len,
                    d,
                ),
            }
            .into());
        }

        for t in 0..seq_len {
            let pos = position_offset + t;
            let emb_start = t * d;
            let pos_start = pos * d;
            for i in 0..d {
                embeddings[emb_start + i] += self.weight[pos_start + i];
            }
        }
        Ok(())
    }
}

// ── EmbeddingNorm ────────────────────────────────────────────────

/// Optional RMS normalization applied after embedding lookup.
///
/// Normalizes each token's embedding vector independently:
/// `output[i] = input[i] / sqrt(mean(input^2) + eps)`
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingNorm {
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Small constant for numerical stability.
    pub eps: f32,
}

impl EmbeddingNorm {
    /// Create a new embedding normalization layer.
    pub fn new(embedding_dim: usize, eps: f32) -> Self {
        Self { embedding_dim, eps }
    }

    /// Normalize embeddings in-place. `data` has shape `[n_tokens, embedding_dim]`.
    pub fn normalize(&self, data: &mut [f32], n_tokens: usize) -> Result<()> {
        let d = self.embedding_dim;
        if data.len() < n_tokens * d {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "data length {} < n_tokens({}) * embedding_dim({})",
                    data.len(),
                    n_tokens,
                    d,
                ),
            }
            .into());
        }

        for t in 0..n_tokens {
            let start = t * d;
            let slice = &mut data[start..start + d];
            let sum_sq: f32 = slice.iter().map(|&v| v * v).sum();
            let scale = 1.0 / (sum_sq / d as f32 + self.eps).sqrt();
            for v in slice.iter_mut() {
                *v *= scale;
            }
        }
        Ok(())
    }
}

// ── CPU reference: embedding lookup ──────────────────────────────

/// Look up embeddings for token IDs (CPU reference).
///
/// For each token ID, copies the corresponding row from `weight` into
/// `output`. Out-of-vocabulary IDs (`>= vocab_size`) produce a zero
/// vector. Tokens matching `padding_idx` also produce zeros.
///
/// # Layout
///
/// * `weight`: `[vocab_size, embedding_dim]` row-major
/// * `output`: `[seq_len, embedding_dim]` row-major (written)
pub fn embedding_lookup_ref(
    token_ids: &[u32],
    weight: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    embedding_dim: usize,
    padding_idx: Option<u32>,
) -> Result<()> {
    let seq_len = token_ids.len();
    if weight.len() < vocab_size * embedding_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight length {} < vocab_size({}) * embedding_dim({})",
                weight.len(),
                vocab_size,
                embedding_dim,
            ),
        }
        .into());
    }
    if output.len() < seq_len * embedding_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output length {} < seq_len({}) * embedding_dim({})",
                output.len(),
                seq_len,
                embedding_dim,
            ),
        }
        .into());
    }

    for (t, &tok) in token_ids.iter().enumerate() {
        let tid = tok as usize;
        let out_start = t * embedding_dim;
        let is_padding = padding_idx.is_some_and(|p| tok == p);

        if tid < vocab_size && !is_padding {
            let src_start = tid * embedding_dim;
            output[out_start..out_start + embedding_dim]
                .copy_from_slice(&weight[src_start..src_start + embedding_dim]);
        } else {
            output[out_start..out_start + embedding_dim].fill(0.0);
        }
    }
    Ok(())
}

// ── CPU reference: output projection ─────────────────────────────

/// Output projection: hidden → logits (CPU reference).
///
/// Computes `output = hidden @ weight^T` where:
/// * `hidden`: `[seq_len, hidden_size]`
/// * `weight`: `[vocab_size, hidden_size]`
/// * `output`: `[seq_len, vocab_size]`
pub fn output_projection_ref(
    hidden: &[f32],
    weight: &[f32],
    output: &mut [f32],
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<()> {
    if hidden.len() < seq_len * hidden_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "hidden length {} < seq_len({}) * hidden_size({})",
                hidden.len(),
                seq_len,
                hidden_size,
            ),
        }
        .into());
    }
    if weight.len() < vocab_size * hidden_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight length {} < vocab_size({}) * hidden_size({})",
                weight.len(),
                vocab_size,
                hidden_size,
            ),
        }
        .into());
    }
    if output.len() < seq_len * vocab_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output length {} < seq_len({}) * vocab_size({})",
                output.len(),
                seq_len,
                vocab_size,
            ),
        }
        .into());
    }

    for s in 0..seq_len {
        for v in 0..vocab_size {
            let mut acc = 0.0f32;
            let h_off = s * hidden_size;
            let w_off = v * hidden_size;
            for k in 0..hidden_size {
                acc += hidden[h_off + k] * weight[w_off + k];
            }
            output[s * vocab_size + v] = acc;
        }
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OpenCL kernel source validation ──────────────────────

    #[test]
    fn opencl_source_is_not_empty() {
        assert!(!EMBEDDING_CL.is_empty(), "embedding.cl should not be empty");
    }

    #[test]
    fn opencl_source_contains_kernel_keyword() {
        assert!(EMBEDDING_CL.contains("__kernel"), "embedding.cl missing __kernel");
    }

    #[test]
    fn opencl_source_has_embedding_lookup_kernel() {
        assert!(
            EMBEDDING_CL.contains("embedding_lookup"),
            "missing embedding_lookup kernel"
        );
    }

    #[test]
    fn opencl_source_has_output_projection_kernel() {
        assert!(
            EMBEDDING_CL.contains("output_projection"),
            "missing output_projection kernel"
        );
    }

    #[test]
    fn opencl_source_has_embedding_rms_norm_kernel() {
        assert!(
            EMBEDDING_CL.contains("embedding_rms_norm"),
            "missing embedding_rms_norm kernel"
        );
    }

    #[test]
    fn opencl_source_has_add_position_embedding_kernel() {
        assert!(
            EMBEDDING_CL.contains("add_position_embedding"),
            "missing add_position_embedding kernel"
        );
    }

    #[test]
    fn opencl_source_has_padded_lookup_kernel() {
        assert!(
            EMBEDDING_CL.contains("embedding_lookup_padded"),
            "missing embedding_lookup_padded kernel"
        );
    }

    // ── EmbeddingConfig ──────────────────────────────────────

    #[test]
    fn config_basic() {
        let cfg = EmbeddingConfig::new(32000, 2048);
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.embedding_dim, 2048);
        assert!(cfg.padding_idx.is_none());
    }

    #[test]
    fn config_with_padding() {
        let cfg = EmbeddingConfig::new(100, 64).with_padding_idx(0);
        assert_eq!(cfg.padding_idx, Some(0));
    }

    // ── EmbeddingTable ───────────────────────────────────────

    #[test]
    fn table_rejects_wrong_weight_size() {
        let cfg = EmbeddingConfig::new(4, 3);
        assert!(EmbeddingTable::new(vec![0.0; 10], cfg).is_err());
    }

    #[test]
    fn table_lookup_basic() {
        let cfg = EmbeddingConfig::new(4, 3);
        let weight = vec![
            1.0, 2.0, 3.0, // 0
            4.0, 5.0, 6.0, // 1
            7.0, 8.0, 9.0, // 2
            10.0, 11.0, 12.0, // 3
        ];
        let table = EmbeddingTable::new(weight, cfg).unwrap();
        let mut out = vec![0.0; 6];
        table.lookup(&[2, 0], &mut out).unwrap();
        assert_eq!(&out[0..3], &[7.0, 8.0, 9.0]);
        assert_eq!(&out[3..6], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn table_lookup_with_padding() {
        let cfg = EmbeddingConfig::new(3, 2).with_padding_idx(1);
        let weight = vec![
            1.0, 2.0, // 0
            3.0, 4.0, // 1 (padding)
            5.0, 6.0, // 2
        ];
        let table = EmbeddingTable::new(weight, cfg).unwrap();
        let mut out = vec![99.0; 6];
        table.lookup(&[0, 1, 2], &mut out).unwrap();
        assert_eq!(&out[0..2], &[1.0, 2.0]);
        assert_eq!(&out[2..4], &[0.0, 0.0]); // padding → zero
        assert_eq!(&out[4..6], &[5.0, 6.0]);
    }

    // ── embedding_lookup_ref ─────────────────────────────────

    #[test]
    fn lookup_ref_known_token_known_vector() {
        let weight = vec![
            0.1, 0.2, // token 0
            0.3, 0.4, // token 1
            0.5, 0.6, // token 2
        ];
        let mut out = vec![0.0; 2];
        embedding_lookup_ref(&[1], &weight, &mut out, 3, 2, None).unwrap();
        assert_eq!(out, vec![0.3, 0.4]);
    }

    #[test]
    fn lookup_ref_oov_returns_zero() {
        let weight = vec![1.0; 6]; // vocab=3, dim=2
        let mut out = vec![99.0; 2];
        embedding_lookup_ref(&[5], &weight, &mut out, 3, 2, None).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn lookup_ref_oov_u32_max() {
        let weight = vec![1.0; 4]; // vocab=2, dim=2
        let mut out = vec![99.0; 2];
        embedding_lookup_ref(&[u32::MAX], &weight, &mut out, 2, 2, None).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn lookup_ref_padding_idx_zeroes() {
        let weight = vec![
            1.0, 2.0, // 0 (padding)
            3.0, 4.0, // 1
        ];
        let mut out = vec![99.0; 4];
        embedding_lookup_ref(&[0, 1], &weight, &mut out, 2, 2, Some(0)).unwrap();
        assert_eq!(&out[0..2], &[0.0, 0.0]);
        assert_eq!(&out[2..4], &[3.0, 4.0]);
    }

    #[test]
    fn lookup_ref_single_token() {
        let weight = vec![42.0];
        let mut out = vec![0.0];
        embedding_lookup_ref(&[0], &weight, &mut out, 1, 1, None).unwrap();
        assert_eq!(out, vec![42.0]);
    }

    #[test]
    fn lookup_ref_vocab_size_one() {
        let weight = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        embedding_lookup_ref(&[0], &weight, &mut out, 1, 3, None).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn lookup_ref_embedding_dim_one() {
        let weight = vec![10.0, 20.0, 30.0];
        let mut out = vec![0.0; 3];
        embedding_lookup_ref(&[2, 0, 1], &weight, &mut out, 3, 1, None).unwrap();
        assert_eq!(out, vec![30.0, 10.0, 20.0]);
    }

    #[test]
    fn lookup_ref_repeated_tokens() {
        let weight = vec![
            1.0, 2.0, // 0
            3.0, 4.0, // 1
        ];
        let mut out = vec![0.0; 8];
        embedding_lookup_ref(&[1, 1, 0, 1], &weight, &mut out, 2, 2, None).unwrap();
        assert_eq!(&out[0..2], &[3.0, 4.0]);
        assert_eq!(&out[2..4], &[3.0, 4.0]);
        assert_eq!(&out[4..6], &[1.0, 2.0]);
        assert_eq!(&out[6..8], &[3.0, 4.0]);
    }

    #[test]
    fn lookup_ref_same_token_same_vector() {
        let weight: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let mut out = vec![0.0; 12];
        embedding_lookup_ref(&[3, 1, 3], &weight, &mut out, 5, 4, None).unwrap();
        assert_eq!(&out[0..4], &out[8..12]); // token 3 == token 3
    }

    #[test]
    fn lookup_ref_all_oov_zeroed() {
        let weight = vec![99.0; 12]; // vocab=3, dim=4
        let mut out = vec![1.0; 12];
        embedding_lookup_ref(&[100, u32::MAX, 3], &weight, &mut out, 3, 4, None)
            .unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lookup_ref_mixed_valid_and_oov() {
        let weight = vec![
            1.0, 2.0, // 0
            3.0, 4.0, // 1
        ];
        let mut out = vec![0.0; 6];
        embedding_lookup_ref(&[0, 999, 1], &weight, &mut out, 2, 2, None).unwrap();
        assert_eq!(&out[0..2], &[1.0, 2.0]);
        assert_eq!(&out[2..4], &[0.0, 0.0]);
        assert_eq!(&out[4..6], &[3.0, 4.0]);
    }

    #[test]
    fn lookup_ref_rejects_short_weight() {
        let mut out = vec![0.0; 2];
        assert!(embedding_lookup_ref(&[0], &[1.0], &mut out, 2, 2, None).is_err());
    }

    #[test]
    fn lookup_ref_rejects_short_output() {
        let weight = vec![1.0; 4];
        let mut out = vec![0.0; 1]; // too small
        assert!(embedding_lookup_ref(&[0], &weight, &mut out, 2, 2, None).is_err());
    }

    #[test]
    fn lookup_ref_empty_tokens() {
        let weight = vec![1.0; 4];
        let mut out = vec![];
        embedding_lookup_ref(&[], &weight, &mut out, 2, 2, None).unwrap();
    }

    #[test]
    fn lookup_ref_large_batch() {
        let vocab = 100;
        let dim = 64;
        let weight: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let ids: Vec<u32> = (0..vocab as u32).collect();
        let mut out = vec![0.0; vocab * dim];
        embedding_lookup_ref(&ids, &weight, &mut out, vocab, dim, None).unwrap();
        assert_eq!(out, weight);
    }

    // ── output_projection_ref ────────────────────────────────

    #[test]
    fn projection_ref_identity_like() {
        // hidden=[1,2], weight=identity-like → logits = hidden @ I^T
        let hidden = vec![1.0, 0.0]; // seq=1, hidden=2
        let weight = vec![
            1.0, 0.0, // vocab 0
            0.0, 1.0, // vocab 1
        ];
        let mut out = vec![0.0; 2];
        output_projection_ref(&hidden, &weight, &mut out, 1, 2, 2).unwrap();
        assert_eq!(out, vec![1.0, 0.0]);
    }

    #[test]
    fn projection_ref_matmul_correctness() {
        // hidden: [[1,2,3]] (1×3), weight: [[1,0,0],[0,1,0],[0,0,1],[1,1,1]] (4×3)
        // logits: [[1, 2, 3, 6]]
        let hidden = vec![1.0, 2.0, 3.0];
        let weight = vec![
            1.0, 0.0, 0.0, // vocab 0
            0.0, 1.0, 0.0, // vocab 1
            0.0, 0.0, 1.0, // vocab 2
            1.0, 1.0, 1.0, // vocab 3
        ];
        let mut out = vec![0.0; 4];
        output_projection_ref(&hidden, &weight, &mut out, 1, 3, 4).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 6.0]);
    }

    #[test]
    fn projection_ref_multi_seq() {
        // hidden: [[1,0],[0,1]] (2×2), weight: [[1,1],[1,-1]] (2×2)
        // logits: [[1,1],[1,-1]]
        let hidden = vec![1.0, 0.0, 0.0, 1.0];
        let weight = vec![1.0, 1.0, 1.0, -1.0];
        let mut out = vec![0.0; 4];
        output_projection_ref(&hidden, &weight, &mut out, 2, 2, 2).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[3] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn projection_ref_single_element() {
        let hidden = vec![3.0]; // seq=1, hidden=1
        let weight = vec![2.0]; // vocab=1, hidden=1
        let mut out = vec![0.0; 1];
        output_projection_ref(&hidden, &weight, &mut out, 1, 1, 1).unwrap();
        assert_eq!(out, vec![6.0]);
    }

    #[test]
    fn projection_ref_rejects_short_hidden() {
        let weight = vec![1.0; 4];
        let mut out = vec![0.0; 2];
        assert!(
            output_projection_ref(&[1.0], &weight, &mut out, 1, 2, 2).is_err()
        );
    }

    #[test]
    fn projection_ref_rejects_short_weight() {
        let hidden = vec![1.0; 2];
        let mut out = vec![0.0; 2];
        assert!(
            output_projection_ref(&hidden, &[1.0], &mut out, 1, 2, 2).is_err()
        );
    }

    #[test]
    fn projection_ref_rejects_short_output() {
        let hidden = vec![1.0; 2];
        let weight = vec![1.0; 4];
        let mut out = vec![0.0; 1]; // too small
        assert!(
            output_projection_ref(&hidden, &weight, &mut out, 1, 2, 2).is_err()
        );
    }

    #[test]
    fn projection_ref_zero_hidden() {
        let hidden = vec![0.0; 4]; // seq=2, hidden=2
        let weight = vec![1.0; 6]; // vocab=3, hidden=2
        let mut out = vec![99.0; 6];
        output_projection_ref(&hidden, &weight, &mut out, 2, 2, 3).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // ── OutputProjection struct ──────────────────────────────

    #[test]
    fn output_projection_struct_rejects_wrong_size() {
        assert!(OutputProjection::new(vec![0.0; 5], 2, 3).is_err());
    }

    #[test]
    fn output_projection_struct_forward() {
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let proj = OutputProjection::new(weight, 2, 2).unwrap();
        let mut out = vec![0.0; 2];
        proj.forward(&[3.0, 7.0], &mut out, 1).unwrap();
        assert_eq!(out, vec![3.0, 7.0]);
    }

    // ── TiedEmbedding ────────────────────────────────────────

    #[test]
    fn tied_rejects_wrong_size() {
        let cfg = EmbeddingConfig::new(2, 3);
        assert!(TiedEmbedding::new(vec![0.0; 5], cfg).is_err());
    }

    #[test]
    fn tied_lookup_matches_standalone() {
        let weight = vec![
            1.0, 2.0, // 0
            3.0, 4.0, // 1
        ];
        let cfg = EmbeddingConfig::new(2, 2);
        let tied = TiedEmbedding::new(weight.clone(), cfg.clone()).unwrap();
        let table = EmbeddingTable::new(weight, cfg).unwrap();

        let mut out_tied = vec![0.0; 4];
        let mut out_table = vec![0.0; 4];
        tied.lookup(&[0, 1], &mut out_tied).unwrap();
        table.lookup(&[0, 1], &mut out_table).unwrap();
        assert_eq!(out_tied, out_table);
    }

    #[test]
    fn tied_projection_uses_same_weight() {
        let weight = vec![
            1.0, 0.0, // vocab 0
            0.0, 1.0, // vocab 1
        ];
        let cfg = EmbeddingConfig::new(2, 2);
        let tied = TiedEmbedding::new(weight.clone(), cfg).unwrap();

        // Lookup token 0 → [1, 0]
        let mut emb = vec![0.0; 2];
        tied.lookup(&[0], &mut emb).unwrap();
        assert_eq!(emb, vec![1.0, 0.0]);

        // Project [1, 0] back → logits should be [1, 0]
        let mut logits = vec![0.0; 2];
        tied.project(&emb, &mut logits, 1).unwrap();
        assert_eq!(logits, vec![1.0, 0.0]);
    }

    #[test]
    fn tied_weight_accessor() {
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = EmbeddingConfig::new(2, 2);
        let tied = TiedEmbedding::new(weight.clone(), cfg).unwrap();
        assert_eq!(tied.weight(), &weight[..]);
    }

    #[test]
    fn tied_roundtrip_embedding_projection() {
        // Weight = [[1,0,0],[0,1,0],[0,0,1]] (identity 3×3)
        // Lookup token 1 → [0,1,0], project → logits = [0,1,0]
        let weight = vec![
            1.0, 0.0, 0.0, // vocab 0
            0.0, 1.0, 0.0, // vocab 1
            0.0, 0.0, 1.0, // vocab 2
        ];
        let cfg = EmbeddingConfig::new(3, 3);
        let tied = TiedEmbedding::new(weight, cfg).unwrap();

        let mut emb = vec![0.0; 3];
        tied.lookup(&[1], &mut emb).unwrap();
        assert_eq!(emb, vec![0.0, 1.0, 0.0]);

        let mut logits = vec![0.0; 3];
        tied.project(&emb, &mut logits, 1).unwrap();
        assert_eq!(logits, vec![0.0, 1.0, 0.0]);
    }

    // ── PositionEmbedding ────────────────────────────────────

    #[test]
    fn position_embedding_rejects_wrong_size() {
        assert!(PositionEmbedding::new(vec![0.0; 5], 2, 3).is_err());
    }

    #[test]
    fn position_embedding_basic() {
        let pos_weight = vec![
            0.1, 0.2, // pos 0
            0.3, 0.4, // pos 1
        ];
        let pos = PositionEmbedding::new(pos_weight, 2, 2).unwrap();
        let mut emb = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens
        pos.add_to(&mut emb, 2, 0).unwrap();
        assert!((emb[0] - 1.1).abs() < 1e-6);
        assert!((emb[1] - 2.2).abs() < 1e-6);
        assert!((emb[2] - 3.3).abs() < 1e-6);
        assert!((emb[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn position_embedding_with_offset() {
        let pos_weight = vec![
            0.1, 0.2, // pos 0
            0.3, 0.4, // pos 1
            0.5, 0.6, // pos 2
        ];
        let pos = PositionEmbedding::new(pos_weight, 3, 2).unwrap();
        let mut emb = vec![1.0, 2.0]; // 1 token, starting at position 2
        pos.add_to(&mut emb, 1, 2).unwrap();
        assert!((emb[0] - 1.5).abs() < 1e-6);
        assert!((emb[1] - 2.6).abs() < 1e-6);
    }

    #[test]
    fn position_embedding_rejects_overflow() {
        let pos_weight = vec![0.0; 4];
        let pos = PositionEmbedding::new(pos_weight, 2, 2).unwrap();
        let mut emb = vec![0.0; 4];
        assert!(pos.add_to(&mut emb, 2, 1).is_err()); // 1+2 > 2
    }

    #[test]
    fn position_embedding_rejects_short_embeddings() {
        let pos_weight = vec![0.0; 4];
        let pos = PositionEmbedding::new(pos_weight, 2, 2).unwrap();
        let mut emb = vec![0.0; 2]; // too small for 2 tokens
        assert!(pos.add_to(&mut emb, 2, 0).is_err());
    }

    #[test]
    fn position_embedding_correct_positions() {
        // Each position adds a distinct value
        let pos_weight = vec![
            10.0, 20.0, // pos 0
            30.0, 40.0, // pos 1
            50.0, 60.0, // pos 2
        ];
        let pos = PositionEmbedding::new(pos_weight, 3, 2).unwrap();
        let mut emb = vec![0.0; 6]; // 3 tokens, zero base
        pos.add_to(&mut emb, 3, 0).unwrap();
        assert_eq!(&emb[0..2], &[10.0, 20.0]);
        assert_eq!(&emb[2..4], &[30.0, 40.0]);
        assert_eq!(&emb[4..6], &[50.0, 60.0]);
    }

    // ── EmbeddingNorm ────────────────────────────────────────

    #[test]
    fn norm_unit_vector_unchanged() {
        // [1, 0] is already RMS-normalized (rms = sqrt(0.5))
        // Actually rms_norm of [1,0] = [1,0] / sqrt((1+0)/2 + eps) ≈ [1.414, 0]
        let norm = EmbeddingNorm::new(2, 1e-6);
        let mut data = vec![1.0, 0.0];
        norm.normalize(&mut data, 1).unwrap();
        let expected_scale = 1.0 / (0.5f32 + 1e-6).sqrt();
        assert!((data[0] - expected_scale).abs() < 1e-4);
        assert!(data[1].abs() < 1e-6);
    }

    #[test]
    fn norm_scales_to_unit_rms() {
        let norm = EmbeddingNorm::new(4, 1e-6);
        let mut data = vec![2.0, 2.0, 2.0, 2.0];
        norm.normalize(&mut data, 1).unwrap();
        // After normalization, RMS should ≈ 1.0
        let rms: f32 =
            (data.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn norm_multi_token() {
        let norm = EmbeddingNorm::new(2, 1e-6);
        let mut data = vec![3.0, 4.0, 0.0, 5.0]; // 2 tokens
        norm.normalize(&mut data, 2).unwrap();
        // Each token normalized independently
        let rms0: f32 = (data[0..2].iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
        let rms1: f32 = (data[2..4].iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
        assert!((rms0 - 1.0).abs() < 1e-4);
        assert!((rms1 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn norm_rejects_short_data() {
        let norm = EmbeddingNorm::new(4, 1e-6);
        let mut data = vec![1.0; 3];
        assert!(norm.normalize(&mut data, 1).is_err());
    }

    #[test]
    fn norm_preserves_direction() {
        let norm = EmbeddingNorm::new(3, 1e-6);
        let mut data = vec![1.0, 2.0, 3.0];
        let orig = data.clone();
        norm.normalize(&mut data, 1).unwrap();
        // Ratios should be preserved
        let ratio01_orig = orig[0] / orig[1];
        let ratio01_norm = data[0] / data[1];
        assert!((ratio01_orig - ratio01_norm).abs() < 1e-6);
        let ratio12_orig = orig[1] / orig[2];
        let ratio12_norm = data[1] / data[2];
        assert!((ratio12_orig - ratio12_norm).abs() < 1e-6);
    }

    // ── Integration / property tests ─────────────────────────

    #[test]
    fn full_pipeline_lookup_norm_project() {
        // Embedding → Norm → Output projection
        let weight = vec![
            1.0, 0.0, // vocab 0
            0.0, 1.0, // vocab 1
        ];
        let cfg = EmbeddingConfig::new(2, 2);
        let table = EmbeddingTable::new(weight.clone(), cfg).unwrap();
        let norm = EmbeddingNorm::new(2, 1e-6);
        let proj = OutputProjection::new(weight, 2, 2).unwrap();

        let mut emb = vec![0.0; 2];
        table.lookup(&[1], &mut emb).unwrap(); // [0, 1]
        norm.normalize(&mut emb, 1).unwrap();
        let mut logits = vec![0.0; 2];
        proj.forward(&emb, &mut logits, 1).unwrap();
        // Token 1 should get highest logit at index 1
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn embedding_output_finite() {
        let weight: Vec<f32> = (0..500).map(|i| (i as f32) * 0.01).collect();
        let cfg = EmbeddingConfig::new(10, 50);
        let table = EmbeddingTable::new(weight, cfg).unwrap();
        let ids: Vec<u32> = (0..10).collect();
        let mut out = vec![f32::NAN; 500];
        table.lookup(&ids, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn projection_output_finite() {
        let hidden: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..640).map(|i| (i as f32) * 0.001).collect();
        let mut out = vec![f32::NAN; 10];
        output_projection_ref(&hidden, &weight, &mut out, 1, 64, 10).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tied_embedding_symmetry() {
        // For identity weight: lookup(t) then project should peak at t
        let weight = vec![
            1.0, 0.0, 0.0, 0.0, // vocab 0
            0.0, 1.0, 0.0, 0.0, // vocab 1
            0.0, 0.0, 1.0, 0.0, // vocab 2
            0.0, 0.0, 0.0, 1.0, // vocab 3
        ];
        let cfg = EmbeddingConfig::new(4, 4);
        let tied = TiedEmbedding::new(weight, cfg).unwrap();

        for token_id in 0..4u32 {
            let mut emb = vec![0.0; 4];
            tied.lookup(&[token_id], &mut emb).unwrap();
            let mut logits = vec![0.0; 4];
            tied.project(&emb, &mut logits, 1).unwrap();
            // Argmax of logits should be token_id
            let argmax = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(argmax, token_id as usize);
        }
    }
}
