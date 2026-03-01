//! OpenCL embedding lookup for A770 GPU inference.
//!
//! Provides:
//! - [`EmbeddingConfig`]: vocabulary/dimension configuration with validation
//! - [`EmbeddingTable`]: weight storage with single-token, batch, and zero-copy lookup
//! - [`EmbeddingError`]: typed errors for out-of-range tokens, dimension mismatches
//! - OpenCL kernel source for GPU-accelerated embedding lookup
//! - [`cpu_embedding_lookup`]: CPU reference implementation for testing

use std::fmt;

// ── Error type ───────────────────────────────────────────────────

/// Errors from embedding operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingError {
    /// Token ID exceeds vocabulary size.
    TokenOutOfRange { token_id: u32, vocab_size: usize },
    /// Weight buffer length does not match config.
    DimensionMismatch { expected: usize, got: usize },
    /// Pre-allocated output buffer is too small.
    OutputBufferTooSmall { needed: usize, got: usize },
    /// Invalid configuration (zero dimensions, etc.).
    InvalidConfig(String),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenOutOfRange { token_id, vocab_size } => {
                write!(f, "token ID {token_id} out of range for vocab_size {vocab_size}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::OutputBufferTooSmall { needed, got } => {
                write!(f, "output buffer too small: needed {needed}, got {got}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid embedding config: {msg}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

// ── Configuration ────────────────────────────────────────────────

/// Configuration for embedding lookup operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingConfig {
    /// Number of tokens in the vocabulary (e.g. 32000 for LLaMA).
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector (e.g. 2048 or 2560).
    pub embed_dim: usize,
}

impl EmbeddingConfig {
    /// Total size of the embedding table in bytes (f32 elements).
    pub fn table_size_bytes(&self) -> usize {
        self.vocab_size * self.embed_dim * 4
    }

    /// Validate that dimensions are non-zero.
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.vocab_size == 0 {
            return Err(EmbeddingError::InvalidConfig("vocab_size must be > 0".to_string()));
        }
        if self.embed_dim == 0 {
            return Err(EmbeddingError::InvalidConfig("embed_dim must be > 0".to_string()));
        }
        Ok(())
    }
}

// ── EmbeddingTable ───────────────────────────────────────────────

/// Token embedding table: maps token IDs to dense vectors.
///
/// Stores a flattened weight matrix `[vocab_size, embed_dim]` in row-major
/// order and provides single-token, batch, and zero-copy lookup methods.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    /// Weight matrix in row-major layout: `[vocab_size, embed_dim]`.
    pub weights: Vec<f32>,
    /// Configuration.
    pub config: EmbeddingConfig,
}

impl EmbeddingTable {
    /// Create a new embedding table.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::InvalidConfig`] if config has zero dimensions,
    /// or [`EmbeddingError::DimensionMismatch`] if weights length doesn't match.
    pub fn new(config: EmbeddingConfig, weights: Vec<f32>) -> Result<Self, EmbeddingError> {
        config.validate()?;
        let expected = config.vocab_size * config.embed_dim;
        if weights.len() != expected {
            return Err(EmbeddingError::DimensionMismatch { expected, got: weights.len() });
        }
        Ok(Self { weights, config })
    }

    /// Look up the embedding vector for a single token ID.
    ///
    /// Returns a slice of `embed_dim` elements.
    pub fn lookup(&self, token_id: u32) -> Result<&[f32], EmbeddingError> {
        let tid = token_id as usize;
        if tid >= self.config.vocab_size {
            return Err(EmbeddingError::TokenOutOfRange {
                token_id,
                vocab_size: self.config.vocab_size,
            });
        }
        let start = tid * self.config.embed_dim;
        Ok(&self.weights[start..start + self.config.embed_dim])
    }

    /// Batch lookup: returns a flattened `[batch_size, embed_dim]` vector.
    pub fn lookup_batch(&self, token_ids: &[u32]) -> Result<Vec<f32>, EmbeddingError> {
        let mut output = vec![0.0f32; token_ids.len() * self.config.embed_dim];
        self.lookup_batch_into(token_ids, &mut output)?;
        Ok(output)
    }

    /// Zero-copy batch lookup into a pre-allocated buffer.
    ///
    /// `output` must have at least `token_ids.len() * embed_dim` elements.
    pub fn lookup_batch_into(
        &self,
        token_ids: &[u32],
        output: &mut [f32],
    ) -> Result<(), EmbeddingError> {
        let needed = token_ids.len() * self.config.embed_dim;
        if output.len() < needed {
            return Err(EmbeddingError::OutputBufferTooSmall { needed, got: output.len() });
        }
        let d = self.config.embed_dim;
        for (i, &tok) in token_ids.iter().enumerate() {
            let row = self.lookup(tok)?;
            output[i * d..(i + 1) * d].copy_from_slice(row);
        }
        Ok(())
    }
}

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for embedding lookup on GPU.
///
/// Contains two kernels:
/// - `embedding_lookup`: single-batch parallel lookup (1D dispatch over embed_dim)
/// - `embedding_lookup_batch`: batch lookup (2D dispatch: batch × embed_dim)
pub const OPENCL_EMBEDDING_SOURCE: &str = r#"
// Embedding lookup kernel for A770 GPU inference.
//
// Memory access pattern: each work-item reads one element from the embedding
// weight table. For a given token ID, consecutive work-items read consecutive
// float values from the same row — this produces coalesced global memory reads
// on GPU architectures (including Intel Xe), maximizing memory bandwidth.

/// Single-token embedding lookup.
/// Global work size: [embed_dim] (one work-item per element).
/// Each work-item copies weight[token_id * embed_dim + gid] to output[gid].
__kernel void embedding_lookup(
    __global const float* weights,   // [vocab_size, embed_dim]
    __global float* output,          // [embed_dim]
    const uint token_id,
    const uint embed_dim,
    const uint vocab_size
) {
    const uint gid = get_global_id(0);
    if (gid >= embed_dim) return;

    if (token_id < vocab_size) {
        // Coalesced read: adjacent work-items read adjacent floats
        output[gid] = weights[token_id * embed_dim + gid];
    } else {
        output[gid] = 0.0f;
    }
}

/// Batch embedding lookup.
/// Global work size: [batch_size, embed_dim] (2D dispatch).
/// dim 0 = batch index, dim 1 = element within embedding vector.
/// Each work-item copies one element of one token's embedding.
__kernel void embedding_lookup_batch(
    __global const float* weights,      // [vocab_size, embed_dim]
    __global const uint* token_ids,     // [batch_size]
    __global float* output,             // [batch_size, embed_dim]
    const uint embed_dim,
    const uint vocab_size
) {
    const uint batch_idx = get_global_id(0);
    const uint dim_idx = get_global_id(1);
    if (dim_idx >= embed_dim) return;

    const uint token_id = token_ids[batch_idx];
    const uint out_offset = batch_idx * embed_dim + dim_idx;

    if (token_id < vocab_size) {
        // Coalesced read: work-items in the same batch row read consecutive
        // floats from the weight table row for token_id.
        output[out_offset] = weights[token_id * embed_dim + dim_idx];
    } else {
        output[out_offset] = 0.0f;
    }
}
"#;

// ── CPU reference implementation ─────────────────────────────────

/// CPU reference implementation of embedding lookup for testing.
///
/// Looks up each token ID in the flat weight table and returns a flattened
/// `[batch_size, embed_dim]` result vector. Out-of-range tokens produce zero
/// vectors.
pub fn cpu_embedding_lookup(
    weights: &[f32],
    vocab_size: usize,
    embed_dim: usize,
    token_ids: &[u32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; token_ids.len() * embed_dim];
    for (i, &tok) in token_ids.iter().enumerate() {
        let tid = tok as usize;
        if tid < vocab_size {
            let src = &weights[tid * embed_dim..(tid + 1) * embed_dim];
            output[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(src);
        }
        // out-of-range tokens remain zero (already initialized)
    }
    output
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple embedding table with sequential weights
    fn make_table(vocab_size: usize, embed_dim: usize) -> EmbeddingTable {
        let weights: Vec<f32> = (0..vocab_size * embed_dim).map(|i| (i + 1) as f32).collect();
        let config = EmbeddingConfig { vocab_size, embed_dim };
        EmbeddingTable::new(config, weights).unwrap()
    }

    // ── EmbeddingConfig tests ────────────────────────────────

    #[test]
    fn config_table_size_bytes() {
        let cfg = EmbeddingConfig { vocab_size: 32000, embed_dim: 2048 };
        assert_eq!(cfg.table_size_bytes(), 32000 * 2048 * 4);
    }

    #[test]
    fn config_table_size_bytes_small() {
        let cfg = EmbeddingConfig { vocab_size: 4, embed_dim: 3 };
        assert_eq!(cfg.table_size_bytes(), 48);
    }

    #[test]
    fn config_validate_ok() {
        let cfg = EmbeddingConfig { vocab_size: 100, embed_dim: 64 };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validate_zero_vocab() {
        let cfg = EmbeddingConfig { vocab_size: 0, embed_dim: 64 };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, EmbeddingError::InvalidConfig(_)));
        assert!(err.to_string().contains("vocab_size"));
    }

    #[test]
    fn config_validate_zero_embed_dim() {
        let cfg = EmbeddingConfig { vocab_size: 100, embed_dim: 0 };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, EmbeddingError::InvalidConfig(_)));
        assert!(err.to_string().contains("embed_dim"));
    }

    #[test]
    fn config_validate_both_zero() {
        let cfg = EmbeddingConfig { vocab_size: 0, embed_dim: 0 };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_clone_eq() {
        let cfg = EmbeddingConfig { vocab_size: 10, embed_dim: 5 };
        assert_eq!(cfg, cfg);
    }

    // ── EmbeddingError tests ─────────────────────────────────

    #[test]
    fn error_display_token_out_of_range() {
        let e = EmbeddingError::TokenOutOfRange { token_id: 50000, vocab_size: 32000 };
        let s = e.to_string();
        assert!(s.contains("50000"));
        assert!(s.contains("32000"));
    }

    #[test]
    fn error_display_dimension_mismatch() {
        let e = EmbeddingError::DimensionMismatch { expected: 100, got: 99 };
        let s = e.to_string();
        assert!(s.contains("100"));
        assert!(s.contains("99"));
    }

    #[test]
    fn error_display_buffer_too_small() {
        let e = EmbeddingError::OutputBufferTooSmall { needed: 256, got: 128 };
        let s = e.to_string();
        assert!(s.contains("256"));
        assert!(s.contains("128"));
    }

    #[test]
    fn error_display_invalid_config() {
        let e = EmbeddingError::InvalidConfig("bad".to_string());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(EmbeddingError::InvalidConfig("test".to_string()));
        assert!(!e.to_string().is_empty());
    }

    // ── EmbeddingTable construction ──────────────────────────

    #[test]
    fn table_new_ok() {
        let config = EmbeddingConfig { vocab_size: 4, embed_dim: 3 };
        let weights = vec![0.0f32; 12];
        assert!(EmbeddingTable::new(config, weights).is_ok());
    }

    #[test]
    fn table_new_dimension_mismatch() {
        let config = EmbeddingConfig { vocab_size: 4, embed_dim: 3 };
        let err = EmbeddingTable::new(config, vec![0.0; 10]).unwrap_err();
        assert!(matches!(err, EmbeddingError::DimensionMismatch { expected: 12, got: 10 }));
    }

    #[test]
    fn table_new_invalid_config_zero_vocab() {
        let config = EmbeddingConfig { vocab_size: 0, embed_dim: 3 };
        assert!(EmbeddingTable::new(config, vec![]).is_err());
    }

    #[test]
    fn table_new_invalid_config_zero_dim() {
        let config = EmbeddingConfig { vocab_size: 4, embed_dim: 0 };
        assert!(EmbeddingTable::new(config, vec![]).is_err());
    }

    // ── Single token lookup ──────────────────────────────────

    #[test]
    fn lookup_first_token() {
        let table = make_table(4, 3);
        let row = table.lookup(0).unwrap();
        assert_eq!(row, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn lookup_last_token() {
        let table = make_table(4, 3);
        let row = table.lookup(3).unwrap();
        assert_eq!(row, &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn lookup_middle_token() {
        let table = make_table(4, 3);
        let row = table.lookup(2).unwrap();
        assert_eq!(row, &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn lookup_out_of_range() {
        let table = make_table(4, 3);
        let err = table.lookup(4).unwrap_err();
        assert!(matches!(err, EmbeddingError::TokenOutOfRange { token_id: 4, vocab_size: 4 }));
    }

    #[test]
    fn lookup_u32_max_out_of_range() {
        let table = make_table(4, 3);
        assert!(table.lookup(u32::MAX).is_err());
    }

    #[test]
    fn lookup_boundary_token_zero() {
        let table = make_table(4, 3);
        assert!(table.lookup(0).is_ok());
    }

    #[test]
    fn lookup_boundary_token_last() {
        let table = make_table(4, 3);
        assert!(table.lookup(3).is_ok());
        assert!(table.lookup(4).is_err());
    }

    #[test]
    fn lookup_embed_dim_one() {
        let config = EmbeddingConfig { vocab_size: 3, embed_dim: 1 };
        let weights = vec![10.0, 20.0, 30.0];
        let table = EmbeddingTable::new(config, weights).unwrap();
        assert_eq!(table.lookup(0).unwrap(), &[10.0]);
        assert_eq!(table.lookup(1).unwrap(), &[20.0]);
        assert_eq!(table.lookup(2).unwrap(), &[30.0]);
    }

    // ── Batch lookup ─────────────────────────────────────────

    #[test]
    fn batch_lookup_single_token() {
        let table = make_table(4, 3);
        let result = table.lookup_batch(&[1]).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn batch_lookup_multiple_tokens() {
        let table = make_table(4, 3);
        let result = table.lookup_batch(&[0, 2]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn batch_lookup_repeated_tokens() {
        let table = make_table(4, 3);
        let result = table.lookup_batch(&[1, 1, 1]).unwrap();
        assert_eq!(&result[0..3], &result[3..6]);
        assert_eq!(&result[0..3], &result[6..9]);
    }

    #[test]
    fn batch_lookup_all_tokens() {
        let table = make_table(4, 3);
        let result = table.lookup_batch(&[0, 1, 2, 3]).unwrap();
        assert_eq!(result, table.weights);
    }

    #[test]
    fn batch_lookup_empty() {
        let table = make_table(4, 3);
        let result = table.lookup_batch(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn batch_lookup_out_of_range() {
        let table = make_table(4, 3);
        let err = table.lookup_batch(&[0, 100]).unwrap_err();
        assert!(matches!(err, EmbeddingError::TokenOutOfRange { token_id: 100, .. }));
    }

    // ── Pre-allocated buffer lookup ──────────────────────────

    #[test]
    fn batch_into_correct_size() {
        let table = make_table(4, 3);
        let mut buf = vec![0.0f32; 6];
        table.lookup_batch_into(&[0, 3], &mut buf).unwrap();
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn batch_into_oversized_buffer() {
        let table = make_table(4, 3);
        let mut buf = vec![99.0f32; 10];
        table.lookup_batch_into(&[1], &mut buf).unwrap();
        assert_eq!(&buf[0..3], &[4.0, 5.0, 6.0]);
        // Trailing elements unchanged
        assert_eq!(buf[3], 99.0);
    }

    #[test]
    fn batch_into_too_small() {
        let table = make_table(4, 3);
        let mut buf = vec![0.0f32; 2]; // need 3
        let err = table.lookup_batch_into(&[0], &mut buf).unwrap_err();
        assert!(matches!(err, EmbeddingError::OutputBufferTooSmall { needed: 3, got: 2 }));
    }

    #[test]
    fn batch_into_empty() {
        let table = make_table(4, 3);
        let mut buf = vec![];
        table.lookup_batch_into(&[], &mut buf).unwrap();
    }

    // ── Large batch ──────────────────────────────────────────

    #[test]
    fn large_batch_correctness() {
        let vocab = 1000;
        let dim = 64;
        let table = make_table(vocab, dim);
        let ids: Vec<u32> = (0..vocab as u32).collect();
        let result = table.lookup_batch(&ids).unwrap();
        assert_eq!(result.len(), vocab * dim);
        // Spot-check first and last
        assert_eq!(result[0], 1.0);
        assert_eq!(result[vocab * dim - 1], (vocab * dim) as f32);
    }

    // ── OpenCL kernel source validation ──────────────────────

    #[test]
    fn opencl_source_not_empty() {
        assert!(!OPENCL_EMBEDDING_SOURCE.is_empty());
    }

    #[test]
    fn opencl_source_has_kernel_keyword() {
        assert!(OPENCL_EMBEDDING_SOURCE.contains("__kernel"));
    }

    #[test]
    fn opencl_source_has_embedding_lookup() {
        assert!(OPENCL_EMBEDDING_SOURCE.contains("embedding_lookup"));
    }

    #[test]
    fn opencl_source_has_batch_kernel() {
        assert!(OPENCL_EMBEDDING_SOURCE.contains("embedding_lookup_batch"));
    }

    #[test]
    fn opencl_source_has_coalesced_comment() {
        assert!(OPENCL_EMBEDDING_SOURCE.contains("coalesced"));
    }

    #[test]
    fn opencl_source_has_get_global_id() {
        assert!(OPENCL_EMBEDDING_SOURCE.contains("get_global_id"));
    }

    // ── CPU reference ────────────────────────────────────────

    #[test]
    fn cpu_ref_matches_table_lookup() {
        let table = make_table(4, 3);
        let ids = [0u32, 1, 2, 3];
        let structured = table.lookup_batch(&ids).unwrap();
        let reference = cpu_embedding_lookup(&table.weights, 4, 3, &ids);
        assert_eq!(structured, reference);
    }

    #[test]
    fn cpu_ref_oov_produces_zeros() {
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let result = cpu_embedding_lookup(&weights, 2, 2, &[5]);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    #[test]
    fn cpu_ref_empty_batch() {
        let weights = vec![1.0; 4];
        let result = cpu_embedding_lookup(&weights, 2, 2, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn cpu_ref_single_element_embedding() {
        let weights = vec![42.0, 99.0];
        let result = cpu_embedding_lookup(&weights, 2, 1, &[0, 1]);
        assert_eq!(result, vec![42.0, 99.0]);
    }

    #[test]
    fn cpu_ref_mixed_valid_and_oov() {
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let result = cpu_embedding_lookup(&weights, 2, 2, &[0, 999, 1]);
        assert_eq!(&result[0..2], &[1.0, 2.0]);
        assert_eq!(&result[2..4], &[0.0, 0.0]);
        assert_eq!(&result[4..6], &[3.0, 4.0]);
    }
}
