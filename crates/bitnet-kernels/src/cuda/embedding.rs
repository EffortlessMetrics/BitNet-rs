//! CUDA embedding lookup kernel with position embedding support.
//!
//! # Kernel strategy
//!
//! Embedding lookup is a memory-bound gather operation: for each token ID in
//! the input sequence, copy the corresponding row from the embedding table.
//!
//! **Token embedding** — one thread-block per token, threads cooperatively
//! copying `embedding_dim` floats from the table row to the output buffer.
//! Grid size equals `seq_len`; block size is `min(embedding_dim, 1024)`.
//!
//! **Position embedding** — a fused second pass that element-wise adds a
//! learned position vector `pos_table[pos]` to each token embedding.  When
//! position embeddings are provided the kernel performs
//! `output[i] = token_table[token_id] + pos_table[position]`, avoiding a
//! separate launch.
//!
//! # CPU fallback
//!
//! [`embedding_lookup_cpu`] and [`embedding_with_position_cpu`] provide
//! equivalent pure-Rust implementations for correctness testing and
//! non-GPU environments.

use bitnet_common::{BitNetError, KernelError, Result};

// ───────────────────────────────────────────────────────────────────
// Launch configuration
// ───────────────────────────────────────────────────────────────────

/// Launch configuration for the embedding lookup kernel.
#[derive(Debug, Clone)]
pub struct EmbeddingKernelConfig {
    /// Number of entries (rows) in the token embedding table.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embedding_dim: usize,
    /// Number of tokens in the input sequence.
    pub seq_len: usize,
    /// Threads per block — typically `min(embedding_dim, 1024)`.
    pub threads_per_block: u32,
    /// Optional padding index whose embedding is always zeros.
    pub padding_idx: Option<u32>,
}

impl EmbeddingKernelConfig {
    /// Create a configuration for the given vocabulary and sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(vocab_size: usize, embedding_dim: usize, seq_len: usize) -> Result<Self> {
        if vocab_size == 0 || embedding_dim == 0 || seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "embedding dimensions must be non-zero: \
                     vocab_size={vocab_size}, embedding_dim={embedding_dim}, \
                     seq_len={seq_len}"
                ),
            }
            .into());
        }
        let threads_per_block = (embedding_dim as u32).min(1024);
        Ok(Self { vocab_size, embedding_dim, seq_len, threads_per_block, padding_idx: None })
    }

    /// Set the padding index.
    pub fn with_padding_idx(mut self, idx: u32) -> Self {
        self.padding_idx = Some(idx);
        self
    }

    /// Compute the CUDA grid dimensions `(seq_len, 1, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.seq_len as u32, 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Configuration for the position embedding addition pass.
#[derive(Debug, Clone)]
pub struct PositionEmbeddingConfig {
    /// Maximum sequence length supported by the position table.
    pub max_seq_len: usize,
    /// Embedding dimensionality (must match token embedding dim).
    pub embedding_dim: usize,
    /// Current sequence length.
    pub seq_len: usize,
    /// Position offset (e.g. for KV-cache continuation).
    pub position_offset: usize,
}

impl PositionEmbeddingConfig {
    /// Create a position embedding configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `embedding_dim` or `seq_len` is zero, or
    /// if `position_offset + seq_len` exceeds `max_seq_len`.
    pub fn new(max_seq_len: usize, embedding_dim: usize, seq_len: usize) -> Result<Self> {
        if embedding_dim == 0 || seq_len == 0 || max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "position embedding dimensions must be non-zero: \
                     max_seq_len={max_seq_len}, \
                     embedding_dim={embedding_dim}, \
                     seq_len={seq_len}"
                ),
            }
            .into());
        }
        if seq_len > max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "seq_len ({seq_len}) exceeds \
                     max_seq_len ({max_seq_len})"
                ),
            }
            .into());
        }
        Ok(Self { max_seq_len, embedding_dim, seq_len, position_offset: 0 })
    }

    /// Set the position offset for KV-cache decode.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset + seq_len > max_seq_len`.
    pub fn with_offset(mut self, offset: usize) -> Result<Self> {
        if offset + self.seq_len > self.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "position_offset ({offset}) + seq_len ({}) \
                     exceeds max_seq_len ({})",
                    self.seq_len, self.max_seq_len
                ),
            }
            .into());
        }
        self.position_offset = offset;
        Ok(self)
    }
}

// ───────────────────────────────────────────────────────────────────
// Error helpers
// ───────────────────────────────────────────────────────────────────

fn index_oob(idx: u32, vocab_size: usize) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments {
        reason: format!(
            "embedding index {idx} out of bounds for \
             vocab_size {vocab_size}"
        ),
    })
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback — token embedding lookup
// ───────────────────────────────────────────────────────────────────

/// Pure-Rust embedding lookup (CPU fallback).
///
/// Copies each `table[token_ids[i]]` row into a contiguous output
/// buffer of shape `[seq_len, embedding_dim]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if any token ID exceeds
/// the vocabulary or the table length is inconsistent.
pub fn embedding_lookup_cpu(
    table: &[f32],
    token_ids: &[u32],
    config: &EmbeddingKernelConfig,
) -> Result<Vec<f32>> {
    let dim = config.embedding_dim;
    if table.len() < config.vocab_size * dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "table length {} < vocab_size ({}) * dim ({})",
                table.len(),
                config.vocab_size,
                dim,
            ),
        }
        .into());
    }
    if token_ids.len() < config.seq_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("token_ids length {} < seq_len ({})", token_ids.len(), config.seq_len,),
        }
        .into());
    }

    let mut output = vec![0.0_f32; config.seq_len * dim];

    for (i, &id) in token_ids.iter().take(config.seq_len).enumerate() {
        if Some(id) == config.padding_idx {
            continue; // already zeroed
        }
        if (id as usize) >= config.vocab_size {
            return Err(index_oob(id, config.vocab_size));
        }
        let src = (id as usize) * dim;
        let dst = i * dim;
        output[dst..dst + dim].copy_from_slice(&table[src..src + dim]);
    }
    Ok(output)
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback — position embedding
// ───────────────────────────────────────────────────────────────────

/// Add learned position embeddings to token embeddings (CPU).
///
/// For each position `p` in `[0, seq_len)`:
///   `output[p * dim .. (p+1) * dim] += pos_table[(p + offset) * dim ..]`
///
/// Operates in-place on `embeddings`.
///
/// # Errors
///
/// Returns an error if buffer sizes are inconsistent.
pub fn embedding_with_position_cpu(
    embeddings: &mut [f32],
    pos_table: &[f32],
    config: &PositionEmbeddingConfig,
) -> Result<()> {
    let dim = config.embedding_dim;
    let required_emb = config.seq_len * dim;
    if embeddings.len() < required_emb {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "embeddings length {} < seq_len ({}) * dim ({})",
                embeddings.len(),
                config.seq_len,
                dim,
            ),
        }
        .into());
    }
    let max_pos = config.position_offset + config.seq_len;
    let required_pos = max_pos * dim;
    if pos_table.len() < required_pos {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "pos_table length {} < \
                 (offset ({}) + seq_len ({})) * dim ({})",
                pos_table.len(),
                config.position_offset,
                config.seq_len,
                dim,
            ),
        }
        .into());
    }

    for pos in 0..config.seq_len {
        let abs_pos = pos + config.position_offset;
        let emb_start = pos * dim;
        let pos_start = abs_pos * dim;
        for j in 0..dim {
            embeddings[emb_start + j] += pos_table[pos_start + j];
        }
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────
// CUDA launch stubs
// ───────────────────────────────────────────────────────────────────

/// Launch stub for the embedding lookup CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is
/// compiled and loaded.
pub fn launch_embedding_lookup(
    _table: &[f32],
    _token_ids: &[u32],
    _output: &mut [f32],
    config: &EmbeddingKernelConfig,
) -> Result<()> {
    log::debug!(
        "embedding lookup stub: vocab={}, dim={}, seq_len={}, \
         grid={:?}",
        config.vocab_size,
        config.embedding_dim,
        config.seq_len,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "embedding CUDA kernel not yet compiled — \
                 scaffold only"
            .into(),
    }
    .into())
}

/// Launch stub for position embedding addition on GPU.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is
/// compiled and loaded.
pub fn launch_position_embedding(
    _embeddings: &mut [f32],
    _pos_table: &[f32],
    config: &PositionEmbeddingConfig,
) -> Result<()> {
    log::debug!(
        "position embedding stub: max_seq={}, dim={}, seq_len={}, \
         offset={}",
        config.max_seq_len,
        config.embedding_dim,
        config.seq_len,
        config.position_offset,
    );
    Err(KernelError::GpuError {
        reason: "position embedding CUDA kernel not yet compiled — \
                 scaffold only"
            .into(),
    }
    .into())
}

// ───────────────────────────────────────────────────────────────────
// Unified dispatch
// ───────────────────────────────────────────────────────────────────

/// Embedding lookup with automatic GPU → CPU fallback.
pub fn embedding_forward(
    table: &[f32],
    token_ids: &[u32],
    config: &EmbeddingKernelConfig,
) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime() {
            let mut output = vec![0.0_f32; config.seq_len * config.embedding_dim];
            if let Ok(()) = launch_embedding_lookup(table, token_ids, &mut output, config) {
                return Ok(output);
            }
        }
    }
    embedding_lookup_cpu(table, token_ids, config)
}

/// Position embedding addition with automatic GPU → CPU fallback.
pub fn position_embedding_forward(
    embeddings: &mut [f32],
    pos_table: &[f32],
    config: &PositionEmbeddingConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_position_embedding(embeddings, pos_table, config)
        {
            return Ok(());
        }
    }
    embedding_with_position_cpu(embeddings, pos_table, config)
}

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 4-word vocab, dim=3 embedding table.
    fn sample_table() -> Vec<f32> {
        vec![
            1.0, 2.0, 3.0, // idx 0
            4.0, 5.0, 6.0, // idx 1
            7.0, 8.0, 9.0, // idx 2
            10.0, 11.0, 12.0, // idx 3
        ]
    }

    fn sample_pos_table() -> Vec<f32> {
        vec![
            0.1, 0.2, 0.3, // pos 0
            0.4, 0.5, 0.6, // pos 1
            0.7, 0.8, 0.9, // pos 2
            1.0, 1.1, 1.2, // pos 3
        ]
    }

    // ── Config tests ────────────────────────────────────────────

    #[test]
    fn test_embedding_config_new() {
        let cfg = EmbeddingKernelConfig::new(32000, 768, 128).unwrap();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.seq_len, 128);
        assert_eq!(cfg.threads_per_block, 768);
        assert!(cfg.padding_idx.is_none());
    }

    #[test]
    fn test_embedding_config_threads_capped() {
        let cfg = EmbeddingKernelConfig::new(32000, 2048, 1).unwrap();
        assert_eq!(cfg.threads_per_block, 1024);
    }

    #[test]
    fn test_embedding_config_rejects_zero() {
        assert!(EmbeddingKernelConfig::new(0, 768, 1).is_err());
        assert!(EmbeddingKernelConfig::new(32000, 0, 1).is_err());
        assert!(EmbeddingKernelConfig::new(32000, 768, 0).is_err());
    }

    #[test]
    fn test_embedding_config_grid_block() {
        let cfg = EmbeddingKernelConfig::new(100, 64, 10).unwrap();
        assert_eq!(cfg.grid_dim(), (10, 1, 1));
        assert_eq!(cfg.block_dim(), (64, 1, 1));
    }

    #[test]
    fn test_embedding_config_padding_idx() {
        let cfg = EmbeddingKernelConfig::new(100, 64, 10).unwrap().with_padding_idx(0);
        assert_eq!(cfg.padding_idx, Some(0));
    }

    #[test]
    fn test_position_config_new() {
        let cfg = PositionEmbeddingConfig::new(512, 768, 128).unwrap();
        assert_eq!(cfg.max_seq_len, 512);
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.seq_len, 128);
        assert_eq!(cfg.position_offset, 0);
    }

    #[test]
    fn test_position_config_rejects_zero() {
        assert!(PositionEmbeddingConfig::new(0, 768, 1).is_err());
        assert!(PositionEmbeddingConfig::new(512, 0, 1).is_err());
        assert!(PositionEmbeddingConfig::new(512, 768, 0).is_err());
    }

    #[test]
    fn test_position_config_rejects_overflow() {
        assert!(PositionEmbeddingConfig::new(10, 768, 11).is_err());
    }

    #[test]
    fn test_position_config_with_offset() {
        let cfg = PositionEmbeddingConfig::new(512, 768, 100).unwrap().with_offset(400).unwrap();
        assert_eq!(cfg.position_offset, 400);
    }

    #[test]
    fn test_position_config_offset_overflow() {
        let cfg = PositionEmbeddingConfig::new(512, 768, 100).unwrap();
        assert!(cfg.with_offset(413).is_err());
    }

    // ── CPU lookup tests ────────────────────────────────────────

    #[test]
    fn test_cpu_lookup_basic() {
        let table = sample_table();
        let cfg = EmbeddingKernelConfig::new(4, 3, 1).unwrap();
        let out = embedding_lookup_cpu(&table, &[2], &cfg).unwrap();
        assert_eq!(out, &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_cpu_lookup_multiple() {
        let table = sample_table();
        let cfg = EmbeddingKernelConfig::new(4, 3, 3).unwrap();
        let out = embedding_lookup_cpu(&table, &[0, 3, 1], &cfg).unwrap();
        assert_eq!(out, &[1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cpu_lookup_duplicate_ids() {
        let table = sample_table();
        let cfg = EmbeddingKernelConfig::new(4, 3, 3).unwrap();
        let out = embedding_lookup_cpu(&table, &[1, 1, 1], &cfg).unwrap();
        assert_eq!(out, &[4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cpu_lookup_padding_idx() {
        let table = sample_table();
        let cfg = EmbeddingKernelConfig::new(4, 3, 3).unwrap().with_padding_idx(1);
        let out = embedding_lookup_cpu(&table, &[0, 1, 2], &cfg).unwrap();
        assert_eq!(&out[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&out[3..6], &[0.0, 0.0, 0.0]); // padding
        assert_eq!(&out[6..9], &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_cpu_lookup_oob() {
        let table = sample_table();
        let cfg = EmbeddingKernelConfig::new(4, 3, 1).unwrap();
        assert!(embedding_lookup_cpu(&table, &[4], &cfg).is_err());
    }

    #[test]
    fn test_cpu_lookup_short_table() {
        let table = vec![1.0, 2.0]; // too small
        let cfg = EmbeddingKernelConfig::new(4, 3, 1).unwrap();
        assert!(embedding_lookup_cpu(&table, &[0], &cfg).is_err());
    }

    #[test]
    fn test_cpu_lookup_short_ids() {
        let table = sample_table();
        let cfg = EmbeddingKernelConfig::new(4, 3, 5).unwrap();
        // only 2 token ids but seq_len=5
        assert!(embedding_lookup_cpu(&table, &[0, 1], &cfg).is_err());
    }

    // ── CPU position embedding tests ────────────────────────────

    #[test]
    fn test_cpu_position_embedding() {
        let table = sample_table();
        let pos_table = sample_pos_table();
        let emb_cfg = EmbeddingKernelConfig::new(4, 3, 2).unwrap();
        let pos_cfg = PositionEmbeddingConfig::new(4, 3, 2).unwrap();

        let mut emb = embedding_lookup_cpu(&table, &[0, 1], &emb_cfg).unwrap();
        embedding_with_position_cpu(&mut emb, &pos_table, &pos_cfg).unwrap();

        // token[0] + pos[0]: [1+0.1, 2+0.2, 3+0.3]
        assert!((emb[0] - 1.1).abs() < 1e-6);
        assert!((emb[1] - 2.2).abs() < 1e-6);
        assert!((emb[2] - 3.3).abs() < 1e-6);
        // token[1] + pos[1]: [4+0.4, 5+0.5, 6+0.6]
        assert!((emb[3] - 4.4).abs() < 1e-6);
        assert!((emb[4] - 5.5).abs() < 1e-6);
        assert!((emb[5] - 6.6).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_position_with_offset() {
        let pos_table = sample_pos_table();
        let pos_cfg = PositionEmbeddingConfig::new(4, 3, 1).unwrap().with_offset(2).unwrap();

        let mut emb = vec![10.0, 20.0, 30.0];
        embedding_with_position_cpu(&mut emb, &pos_table, &pos_cfg).unwrap();

        // emb + pos[2]: [10+0.7, 20+0.8, 30+0.9]
        assert!((emb[0] - 10.7).abs() < 1e-6);
        assert!((emb[1] - 20.8).abs() < 1e-6);
        assert!((emb[2] - 30.9).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_position_short_embeddings() {
        let pos_table = sample_pos_table();
        let pos_cfg = PositionEmbeddingConfig::new(4, 3, 2).unwrap();
        let mut emb = vec![1.0, 2.0]; // too short
        assert!(embedding_with_position_cpu(&mut emb, &pos_table, &pos_cfg).is_err());
    }

    #[test]
    fn test_cpu_position_short_pos_table() {
        let pos_cfg = PositionEmbeddingConfig::new(4, 3, 2).unwrap();
        let mut emb = vec![0.0; 6];
        let pos_table = vec![0.1, 0.2, 0.3]; // only 1 row
        assert!(embedding_with_position_cpu(&mut emb, &pos_table, &pos_cfg).is_err());
    }

    // ── Unified dispatch tests ──────────────────────────────────

    #[test]
    fn test_forward_dispatches_cpu() {
        let table = sample_table();
        let cfg = EmbeddingKernelConfig::new(4, 3, 2).unwrap();
        let out = embedding_forward(&table, &[1, 3], &cfg).unwrap();
        assert_eq!(out, &[4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_position_forward_dispatches_cpu() {
        let pos_table = sample_pos_table();
        let pos_cfg = PositionEmbeddingConfig::new(4, 3, 2).unwrap();
        let mut emb = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        position_embedding_forward(&mut emb, &pos_table, &pos_cfg).unwrap();
        assert!((emb[0] - 1.1).abs() < 1e-6);
        assert!((emb[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn test_forward_matches_cpu() {
        let table = sample_table();
        let cfg = EmbeddingKernelConfig::new(4, 3, 4).unwrap();
        let ids = [3, 0, 2, 1];

        let fwd = embedding_forward(&table, &ids, &cfg).unwrap();
        let cpu = embedding_lookup_cpu(&table, &ids, &cfg).unwrap();
        assert_eq!(fwd, cpu);
    }

    // ── GPU launch stub tests ───────────────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_embedding_lookup_launch() {
        let table = vec![0.0_f32; 32000 * 768];
        let ids = vec![0_u32; 128];
        let mut output = vec![0.0_f32; 128 * 768];
        let cfg = EmbeddingKernelConfig::new(32000, 768, 128).unwrap();
        let result = launch_embedding_lookup(&table, &ids, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA embedding launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu"]
    fn test_cuda_position_embedding_launch() {
        let pos_table = vec![0.0_f32; 512 * 768];
        let mut emb = vec![0.0_f32; 128 * 768];
        let cfg = PositionEmbeddingConfig::new(512, 768, 128).unwrap();
        let result = launch_position_embedding(&mut emb, &pos_table, &cfg);
        assert!(result.is_ok(), "CUDA position embedding launch failed: {result:?}");
    }
}
