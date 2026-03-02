//! CPU SIMD-friendly gather/scatter operations and embedding lookup.
//!
//! Provides safe Rust equivalents of hardware gather/scatter intrinsics
//! (e.g. AVX2 `_mm256_i32gather_ps`) with automatic bounds checking and
//! optional masking.  The [`EmbeddingLookup`] struct adds cache-friendly
//! batched token-ID → embedding-vector retrieval suitable for transformer
//! input pipelines.
//!
//! All operations are pure-Rust and do not require SIMD feature flags at
//! compile time; however, the data-access patterns are designed so that
//! auto-vectorisation by LLVM produces efficient gather/scatter code on
//! AVX2 and NEON targets.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// GatherOp
// ---------------------------------------------------------------------------

/// Indexed gather from contiguous f32 / i32 arrays.
pub struct GatherOp;

impl GatherOp {
    /// Gather `f32` values from `src` at positions given by `indices`.
    ///
    /// Equivalent to `out[i] = src[indices[i]]` for each valid index.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if any index is out of bounds.
    pub fn gather_f32(src: &[f32], indices: &[u32], out: &mut [f32]) -> Result<()> {
        if out.len() < indices.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < indices length {}", out.len(), indices.len()),
            }
            .into());
        }
        let src_len = src.len() as u32;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= src_len {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "gather index {idx} out of bounds for src length {}",
                        src.len()
                    ),
                }
                .into());
            }
            out[i] = src[idx as usize];
        }
        Ok(())
    }

    /// Gather `i32` values from `src` at positions given by `indices`.
    pub fn gather_i32(src: &[i32], indices: &[u32], out: &mut [i32]) -> Result<()> {
        if out.len() < indices.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < indices length {}", out.len(), indices.len()),
            }
            .into());
        }
        let src_len = src.len() as u32;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= src_len {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "gather index {idx} out of bounds for src length {}",
                        src.len()
                    ),
                }
                .into());
            }
            out[i] = src[idx as usize];
        }
        Ok(())
    }

    /// Masked gather: uses `defaults[i]` when `mask[i]` is `false`.
    ///
    /// For lanes where `mask[i]` is `true`, reads `src[indices[i]]`;
    /// out-of-bounds masked-true lanes return an error.
    pub fn masked_gather_f32(
        src: &[f32],
        indices: &[u32],
        mask: &[bool],
        defaults: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let n = indices.len();
        if mask.len() < n || defaults.len() < n || out.len() < n {
            return Err(KernelError::InvalidArguments {
                reason: "mask, defaults, and output must be at least as long as indices".into(),
            }
            .into());
        }
        let src_len = src.len() as u32;
        for i in 0..n {
            if mask[i] {
                let idx = indices[i];
                if idx >= src_len {
                    return Err(KernelError::InvalidArguments {
                        reason: format!(
                            "masked gather: index {idx} out of bounds for src length {}",
                            src.len()
                        ),
                    }
                    .into());
                }
                out[i] = src[idx as usize];
            } else {
                out[i] = defaults[i];
            }
        }
        Ok(())
    }

    /// Batched gather for embedding lookups: for each token ID in `token_ids`,
    /// copies the corresponding row from `table` (shape `[vocab_size, dim]`)
    /// into `out`.
    ///
    /// `out` must have length `≥ token_ids.len() * dim`.
    pub fn batched_gather(
        table: &[f32],
        vocab_size: usize,
        dim: usize,
        token_ids: &[u32],
        out: &mut [f32],
    ) -> Result<()> {
        let n = token_ids.len();
        if table.len() < vocab_size * dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "table length {} < vocab_size*dim = {}",
                    table.len(),
                    vocab_size * dim,
                ),
            }
            .into());
        }
        if out.len() < n * dim {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < n*dim = {}", out.len(), n * dim),
            }
            .into());
        }
        for (i, &tid) in token_ids.iter().enumerate() {
            if tid as usize >= vocab_size {
                return Err(KernelError::InvalidArguments {
                    reason: format!("token id {tid} out of bounds for vocab size {vocab_size}"),
                }
                .into());
            }
            let src_off = tid as usize * dim;
            let dst_off = i * dim;
            out[dst_off..dst_off + dim].copy_from_slice(&table[src_off..src_off + dim]);
        }
        Ok(())
    }

    /// Strided gather: reads `src[indices[i] * stride]` for non-contiguous
    /// access patterns.
    pub fn strided_gather_f32(
        src: &[f32],
        indices: &[u32],
        stride: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if out.len() < indices.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < indices length {}", out.len(), indices.len()),
            }
            .into());
        }
        if stride == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "stride must be > 0".into() }.into()
            );
        }
        for (i, &idx) in indices.iter().enumerate() {
            let offset = idx as usize * stride;
            if offset >= src.len() {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "strided gather offset {} (index {idx} * stride {stride}) \
                         out of bounds for src length {}",
                        offset,
                        src.len()
                    ),
                }
                .into());
            }
            out[i] = src[offset];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ScatterOp
// ---------------------------------------------------------------------------

/// Indexed scatter to contiguous f32 arrays.
pub struct ScatterOp;

impl ScatterOp {
    /// Scatter `values[i]` into `dst[indices[i]]`.
    ///
    /// When multiple indices map to the same location the last write wins.
    pub fn scatter_f32(values: &[f32], indices: &[u32], dst: &mut [f32]) -> Result<()> {
        let n = values.len();
        if indices.len() < n {
            return Err(KernelError::InvalidArguments {
                reason: format!("indices length {} < values length {}", indices.len(), n),
            }
            .into());
        }
        let dst_len = dst.len() as u32;
        for i in 0..n {
            let idx = indices[i];
            if idx >= dst_len {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "scatter index {idx} out of bounds for dst length {}",
                        dst.len()
                    ),
                }
                .into());
            }
            dst[idx as usize] = values[i];
        }
        Ok(())
    }

    /// Scatter-add: `dst[indices[i]] += values[i]`.
    ///
    /// Accumulates rather than overwrites, matching the semantics of
    /// gradient scatter in embedding layers.
    pub fn scatter_add_f32(values: &[f32], indices: &[u32], dst: &mut [f32]) -> Result<()> {
        let n = values.len();
        if indices.len() < n {
            return Err(KernelError::InvalidArguments {
                reason: format!("indices length {} < values length {}", indices.len(), n),
            }
            .into());
        }
        let dst_len = dst.len() as u32;
        for i in 0..n {
            let idx = indices[i];
            if idx >= dst_len {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "scatter_add index {idx} out of bounds for dst length {}",
                        dst.len()
                    ),
                }
                .into());
            }
            dst[idx as usize] += values[i];
        }
        Ok(())
    }

    /// Masked scatter: only writes when `mask[i]` is `true`.
    pub fn masked_scatter_f32(
        values: &[f32],
        indices: &[u32],
        mask: &[bool],
        dst: &mut [f32],
    ) -> Result<()> {
        let n = values.len();
        if indices.len() < n || mask.len() < n {
            return Err(KernelError::InvalidArguments {
                reason: "indices and mask must be at least as long as values".into(),
            }
            .into());
        }
        let dst_len = dst.len() as u32;
        for i in 0..n {
            if mask[i] {
                let idx = indices[i];
                if idx >= dst_len {
                    return Err(KernelError::InvalidArguments {
                        reason: format!(
                            "masked scatter index {idx} out of bounds for dst length {}",
                            dst.len()
                        ),
                    }
                    .into());
                }
                dst[idx as usize] = values[i];
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EmbeddingLookup
// ---------------------------------------------------------------------------

/// Cache-friendly embedding lookup optimised for transformer input pipelines.
///
/// Stores a reference to a flat embedding table `[vocab_size, embed_dim]` and
/// provides batched, prefetch-hinted lookups.
pub struct EmbeddingLookup<'a> {
    table: &'a [f32],
    vocab_size: usize,
    embed_dim: usize,
}

impl<'a> EmbeddingLookup<'a> {
    /// Create an embedding lookup over `table` with the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if `table.len() < vocab_size * embed_dim`.
    pub fn new(table: &'a [f32], vocab_size: usize, embed_dim: usize) -> Result<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "vocab_size ({vocab_size}) and embed_dim ({embed_dim}) must be > 0"
                ),
            }
            .into());
        }
        if table.len() < vocab_size * embed_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "table length {} < vocab_size*embed_dim = {}",
                    table.len(),
                    vocab_size * embed_dim,
                ),
            }
            .into());
        }
        Ok(Self { table, vocab_size, embed_dim })
    }

    /// Return the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Look up a single token and write its embedding into `out`.
    pub fn lookup(&self, token_id: u32, out: &mut [f32]) -> Result<()> {
        if token_id as usize >= self.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("token_id {token_id} >= vocab_size {}", self.vocab_size),
            }
            .into());
        }
        if out.len() < self.embed_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!("output length {} < embed_dim {}", out.len(), self.embed_dim),
            }
            .into());
        }
        let off = token_id as usize * self.embed_dim;
        out[..self.embed_dim].copy_from_slice(&self.table[off..off + self.embed_dim]);
        Ok(())
    }

    /// Batched lookup: writes embeddings for all `token_ids` into `out`
    /// (row-major, `out` length ≥ `token_ids.len() * embed_dim`).
    ///
    /// Uses a prefetch-friendly access pattern: each subsequent row is
    /// touched before the copy so the hardware prefetcher can stay ahead.
    pub fn batched_lookup(&self, token_ids: &[u32], out: &mut [f32]) -> Result<()> {
        let n = token_ids.len();
        if out.len() < n * self.embed_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < n*embed_dim = {}",
                    out.len(),
                    n * self.embed_dim,
                ),
            }
            .into());
        }
        for (i, &tid) in token_ids.iter().enumerate() {
            if tid as usize >= self.vocab_size {
                return Err(KernelError::InvalidArguments {
                    reason: format!("token_id {tid} >= vocab_size {}", self.vocab_size),
                }
                .into());
            }
            let src_off = tid as usize * self.embed_dim;
            // Hint: touch next row to warm the cache line for the next iteration.
            if i + 1 < n {
                let next_tid = token_ids[i + 1] as usize;
                if next_tid < self.vocab_size {
                    let _prefetch_hint = self.table[next_tid * self.embed_dim];
                    std::hint::black_box(_prefetch_hint);
                }
            }
            let dst_off = i * self.embed_dim;
            out[dst_off..dst_off + self.embed_dim]
                .copy_from_slice(&self.table[src_off..src_off + self.embed_dim]);
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- GatherOp: gather_f32 -----------------------------------------------

    #[test]
    fn test_gather_f32_basic() {
        let src = [10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = [4, 2, 0];
        let mut out = [0.0_f32; 3];
        GatherOp::gather_f32(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [50.0, 30.0, 10.0]);
    }

    #[test]
    fn test_gather_f32_duplicate_indices() {
        let src = [1.0, 2.0, 3.0];
        let indices = [1, 1, 1];
        let mut out = [0.0_f32; 3];
        GatherOp::gather_f32(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_gather_f32_oob() {
        let src = [1.0, 2.0];
        let indices = [5];
        let mut out = [0.0_f32; 1];
        assert!(GatherOp::gather_f32(&src, &indices, &mut out).is_err());
    }

    #[test]
    fn test_gather_f32_empty() {
        let src = [1.0];
        let indices: [u32; 0] = [];
        let mut out: [f32; 0] = [];
        GatherOp::gather_f32(&src, &indices, &mut out).unwrap();
    }

    #[test]
    fn test_gather_f32_output_too_short() {
        let src = [1.0, 2.0, 3.0];
        let indices = [0, 1, 2];
        let mut out = [0.0_f32; 2]; // too short
        assert!(GatherOp::gather_f32(&src, &indices, &mut out).is_err());
    }

    // -- GatherOp: gather_i32 -----------------------------------------------

    #[test]
    fn test_gather_i32_basic() {
        let src = [100, 200, 300, 400];
        let indices = [3, 0, 1];
        let mut out = [0_i32; 3];
        GatherOp::gather_i32(&src, &indices, &mut out).unwrap();
        assert_eq!(out, [400, 100, 200]);
    }

    #[test]
    fn test_gather_i32_oob() {
        let src = [1, 2];
        let indices = [2]; // OOB (len==2, max valid idx=1)
        let mut out = [0_i32; 1];
        assert!(GatherOp::gather_i32(&src, &indices, &mut out).is_err());
    }

    // -- GatherOp: masked_gather_f32 ----------------------------------------

    #[test]
    fn test_masked_gather_all_true() {
        let src = [10.0, 20.0, 30.0];
        let indices = [2, 0, 1];
        let mask = [true, true, true];
        let defaults = [0.0; 3];
        let mut out = [0.0_f32; 3];
        GatherOp::masked_gather_f32(&src, &indices, &mask, &defaults, &mut out).unwrap();
        assert_eq!(out, [30.0, 10.0, 20.0]);
    }

    #[test]
    fn test_masked_gather_mixed() {
        let src = [10.0, 20.0, 30.0];
        let indices = [2, 0, 1];
        let mask = [true, false, true];
        let defaults = [-1.0, -2.0, -3.0];
        let mut out = [0.0_f32; 3];
        GatherOp::masked_gather_f32(&src, &indices, &mask, &defaults, &mut out).unwrap();
        assert_eq!(out, [30.0, -2.0, 20.0]);
    }

    #[test]
    fn test_masked_gather_all_false() {
        let src = [10.0, 20.0];
        let indices = [0, 1];
        let mask = [false, false];
        let defaults = [99.0, 88.0];
        let mut out = [0.0_f32; 2];
        GatherOp::masked_gather_f32(&src, &indices, &mask, &defaults, &mut out).unwrap();
        assert_eq!(out, [99.0, 88.0]);
    }

    #[test]
    fn test_masked_gather_oob_masked_true() {
        let src = [1.0];
        let indices = [5]; // OOB but mask=true → error
        let mask = [true];
        let defaults = [0.0];
        let mut out = [0.0_f32; 1];
        assert!(GatherOp::masked_gather_f32(&src, &indices, &mask, &defaults, &mut out).is_err());
    }

    #[test]
    fn test_masked_gather_oob_masked_false() {
        let src = [1.0];
        let indices = [5]; // OOB but mask=false → use default
        let mask = [false];
        let defaults = [42.0];
        let mut out = [0.0_f32; 1];
        GatherOp::masked_gather_f32(&src, &indices, &mask, &defaults, &mut out).unwrap();
        assert_eq!(out, [42.0]);
    }

    // -- GatherOp: batched_gather -------------------------------------------

    #[test]
    fn test_batched_gather_basic() {
        // 3 tokens, dim=2
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let token_ids = [2, 0];
        let mut out = [0.0_f32; 4];
        GatherOp::batched_gather(&table, 3, 2, &token_ids, &mut out).unwrap();
        assert_eq!(out, [5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn test_batched_gather_oob_token() {
        let table = [1.0, 2.0, 3.0, 4.0];
        let token_ids = [2]; // vocab=2, so id=2 is OOB
        let mut out = [0.0_f32; 2];
        assert!(GatherOp::batched_gather(&table, 2, 2, &token_ids, &mut out).is_err());
    }

    // -- GatherOp: strided_gather_f32 ---------------------------------------

    #[test]
    fn test_strided_gather_basic() {
        let src = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = [0, 2]; // stride=3 → offsets 0, 6 → but 6 OOB
        let mut out = [0.0_f32; 2];
        // stride 2: offsets 0, 4
        let indices2 = [0, 2];
        GatherOp::strided_gather_f32(&src, &indices2, 2, &mut out).unwrap();
        assert_eq!(out, [0.0, 4.0]);
    }

    #[test]
    fn test_strided_gather_oob() {
        let src = [1.0, 2.0, 3.0];
        let indices = [2]; // stride=2 → offset 4, OOB
        let mut out = [0.0_f32; 1];
        assert!(GatherOp::strided_gather_f32(&src, &indices, 2, &mut out).is_err());
    }

    #[test]
    fn test_strided_gather_zero_stride() {
        let src = [1.0];
        let indices = [0];
        let mut out = [0.0_f32; 1];
        assert!(GatherOp::strided_gather_f32(&src, &indices, 0, &mut out).is_err());
    }

    // -- ScatterOp: scatter_f32 ---------------------------------------------

    #[test]
    fn test_scatter_f32_basic() {
        let values = [10.0, 20.0, 30.0];
        let indices = [4, 1, 3];
        let mut dst = [0.0_f32; 5];
        ScatterOp::scatter_f32(&values, &indices, &mut dst).unwrap();
        assert_eq!(dst, [0.0, 20.0, 0.0, 30.0, 10.0]);
    }

    #[test]
    fn test_scatter_f32_oob() {
        let values = [1.0];
        let indices = [5];
        let mut dst = [0.0_f32; 3];
        assert!(ScatterOp::scatter_f32(&values, &indices, &mut dst).is_err());
    }

    #[test]
    fn test_scatter_f32_duplicate_last_wins() {
        let values = [1.0, 2.0, 3.0];
        let indices = [0, 0, 0];
        let mut dst = [0.0_f32; 1];
        ScatterOp::scatter_f32(&values, &indices, &mut dst).unwrap();
        assert_eq!(dst, [3.0]);
    }

    // -- ScatterOp: scatter_add_f32 -----------------------------------------

    #[test]
    fn test_scatter_add_basic() {
        let values = [1.0, 2.0, 3.0];
        let indices = [0, 0, 1];
        let mut dst = [0.0_f32; 2];
        ScatterOp::scatter_add_f32(&values, &indices, &mut dst).unwrap();
        assert_eq!(dst, [3.0, 3.0]);
    }

    #[test]
    fn test_scatter_add_oob() {
        let values = [1.0];
        let indices = [10];
        let mut dst = [0.0_f32; 2];
        assert!(ScatterOp::scatter_add_f32(&values, &indices, &mut dst).is_err());
    }

    // -- ScatterOp: masked_scatter_f32 --------------------------------------

    #[test]
    fn test_masked_scatter_mixed() {
        let values = [10.0, 20.0, 30.0];
        let indices = [0, 1, 2];
        let mask = [true, false, true];
        let mut dst = [0.0_f32; 3];
        ScatterOp::masked_scatter_f32(&values, &indices, &mask, &mut dst).unwrap();
        assert_eq!(dst, [10.0, 0.0, 30.0]);
    }

    #[test]
    fn test_masked_scatter_oob_masked_true() {
        let values = [1.0];
        let indices = [99];
        let mask = [true];
        let mut dst = [0.0_f32; 2];
        assert!(ScatterOp::masked_scatter_f32(&values, &indices, &mask, &mut dst).is_err());
    }

    #[test]
    fn test_masked_scatter_oob_masked_false() {
        let values = [1.0];
        let indices = [99]; // OOB but masked false → skip
        let mask = [false];
        let mut dst = [0.0_f32; 2];
        ScatterOp::masked_scatter_f32(&values, &indices, &mask, &mut dst).unwrap();
        assert_eq!(dst, [0.0, 0.0]);
    }

    // -- EmbeddingLookup ----------------------------------------------------

    fn make_table(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab * dim).map(|i| i as f32).collect()
    }

    #[test]
    fn test_embedding_new_valid() {
        let table = make_table(10, 4);
        let emb = EmbeddingLookup::new(&table, 10, 4).unwrap();
        assert_eq!(emb.vocab_size(), 10);
        assert_eq!(emb.embed_dim(), 4);
    }

    #[test]
    fn test_embedding_new_zero_dim() {
        let table = [0.0; 10];
        assert!(EmbeddingLookup::new(&table, 0, 4).is_err());
        assert!(EmbeddingLookup::new(&table, 4, 0).is_err());
    }

    #[test]
    fn test_embedding_new_table_too_small() {
        let table = [0.0; 5];
        assert!(EmbeddingLookup::new(&table, 3, 4).is_err());
    }

    #[test]
    fn test_embedding_single_lookup() {
        let table = make_table(4, 3);
        let emb = EmbeddingLookup::new(&table, 4, 3).unwrap();
        let mut out = [0.0_f32; 3];
        emb.lookup(2, &mut out).unwrap();
        assert_eq!(out, [6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_embedding_lookup_oob() {
        let table = make_table(4, 3);
        let emb = EmbeddingLookup::new(&table, 4, 3).unwrap();
        let mut out = [0.0_f32; 3];
        assert!(emb.lookup(4, &mut out).is_err());
    }

    #[test]
    fn test_embedding_batched_lookup() {
        let table = make_table(5, 2);
        let emb = EmbeddingLookup::new(&table, 5, 2).unwrap();
        let ids = [0, 3, 1];
        let mut out = [0.0_f32; 6];
        emb.batched_lookup(&ids, &mut out).unwrap();
        assert_eq!(out, [0.0, 1.0, 6.0, 7.0, 2.0, 3.0]);
    }

    #[test]
    fn test_embedding_batched_oob() {
        let table = make_table(3, 2);
        let emb = EmbeddingLookup::new(&table, 3, 2).unwrap();
        let ids = [0, 3]; // 3 OOB
        let mut out = [0.0_f32; 4];
        assert!(emb.batched_lookup(&ids, &mut out).is_err());
    }

    #[test]
    fn test_embedding_batched_output_too_short() {
        let table = make_table(3, 2);
        let emb = EmbeddingLookup::new(&table, 3, 2).unwrap();
        let ids = [0, 1];
        let mut out = [0.0_f32; 3]; // needs 4
        assert!(emb.batched_lookup(&ids, &mut out).is_err());
    }

    // -- Property tests with proptest ---------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn gather_f32_never_panics(
                src in prop::collection::vec(-1e6_f32..1e6, 1..128usize),
                raw_indices in prop::collection::vec(0u32..256, 1..64usize),
            ) {
                let indices: Vec<u32> = raw_indices.iter()
                    .map(|&i| i % src.len() as u32)
                    .collect();
                let mut out = vec![0.0_f32; indices.len()];
                GatherOp::gather_f32(&src, &indices, &mut out).unwrap();
                for (j, &idx) in indices.iter().enumerate() {
                    prop_assert_eq!(out[j], src[idx as usize]);
                }
            }

            #[test]
            fn scatter_add_is_commutative_sum(
                n in 1usize..64,
                dst_len in 1usize..32,
            ) {
                let values: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
                let indices: Vec<u32> = (0..n).map(|i| (i % dst_len) as u32).collect();
                let mut dst = vec![0.0_f32; dst_len];
                ScatterOp::scatter_add_f32(&values, &indices, &mut dst).unwrap();
                let total: f32 = dst.iter().sum();
                let expected: f32 = values.iter().sum();
                prop_assert!((total - expected).abs() < 1e-3,
                    "total {total} != expected {expected}");
            }

            #[test]
            fn masked_gather_defaults_on_false(
                src in prop::collection::vec(0.0_f32..100.0, 1..64usize),
                n in 1usize..32,
            ) {
                let defaults: Vec<f32> = (0..n).map(|i| -(i as f32) - 1.0).collect();
                let indices: Vec<u32> = (0..n).map(|i| (i % src.len()) as u32).collect();
                let mask = vec![false; n];
                let mut out = vec![0.0_f32; n];
                GatherOp::masked_gather_f32(&src, &indices, &mask, &defaults, &mut out)
                    .unwrap();
                for i in 0..n {
                    prop_assert_eq!(out[i], defaults[i]);
                }
            }

            #[test]
            fn embedding_roundtrip(
                vocab in 1usize..32,
                dim in 1usize..64,
            ) {
                let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
                let emb = EmbeddingLookup::new(&table, vocab, dim).unwrap();
                for tid in 0..vocab {
                    let mut out = vec![0.0_f32; dim];
                    emb.lookup(tid as u32, &mut out).unwrap();
                    let expected: Vec<f32> = (tid * dim..(tid + 1) * dim)
                        .map(|i| i as f32)
                        .collect();
                    prop_assert_eq!(out, expected);
                }
            }

            #[test]
            fn strided_gather_correct(
                stride in 1usize..8,
                n in 1usize..16,
            ) {
                let src_len = n * stride;
                let src: Vec<f32> = (0..src_len).map(|i| i as f32).collect();
                let indices: Vec<u32> = (0..n as u32).collect();
                let mut out = vec![0.0_f32; n];
                GatherOp::strided_gather_f32(&src, &indices, stride, &mut out).unwrap();
                for i in 0..n {
                    prop_assert_eq!(out[i], (i * stride) as f32);
                }
            }
        }
    }
}
