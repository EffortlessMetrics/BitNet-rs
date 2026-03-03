//! SIMD-optimized embedding lookup for CPU inference.
//!
//! Provides high-performance token and position embedding operations with
//! AVX2-accelerated gather and arithmetic, falling back to scalar code on
//! platforms without AVX2.
//!
//! # Operations
//!
//! - **Embedding lookup**: single or batched token ID → dense vector.
//! - **Embedding aggregation**: element-wise sum / mean of multiple embeddings.
//! - **Position embedding**: absolute additive position encoding.
//! - **Rotary embedding**: in-place RoPE application on query/key pairs.
//!
//! # SIMD strategy
//!
//! On `x86_64` with AVX2 detected at runtime the hot loops process 8×f32
//! lanes at a time using `_mm256_loadu_ps` / `_mm256_storeu_ps`.  A scalar
//! tail loop handles the remaining elements.  Non-x86 targets always take
//! the scalar path.

use bitnet_common::KernelError;

// ── Error type ───────────────────────────────────────────────────

/// Errors specific to embedding operations.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingError {
    /// A token ID exceeds the vocabulary size.
    TokenOutOfRange { token_id: u32, vocab_size: usize },
    /// The weight buffer has the wrong length.
    WeightShapeMismatch { got: usize, expected: usize },
    /// The output buffer has the wrong length.
    OutputShapeMismatch { got: usize, expected: usize },
    /// A position index exceeds the maximum sequence length.
    PositionOutOfRange { position: usize, max_len: usize },
    /// Embedding dimension must be even for rotary embeddings.
    OddEmbeddingDim { dim: usize },
    /// Generic invalid argument.
    InvalidArgument(String),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenOutOfRange { token_id, vocab_size } => {
                write!(f, "token id {token_id} out of range for vocab size {vocab_size}")
            }
            Self::WeightShapeMismatch { got, expected } => {
                write!(f, "weight length {got} != expected {expected}")
            }
            Self::OutputShapeMismatch { got, expected } => {
                write!(f, "output length {got} != expected {expected}")
            }
            Self::PositionOutOfRange { position, max_len } => {
                write!(f, "position {position} out of range for max length {max_len}")
            }
            Self::OddEmbeddingDim { dim } => {
                write!(f, "embedding dim {dim} must be even for rotary embedding")
            }
            Self::InvalidArgument(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

impl From<EmbeddingError> for bitnet_common::BitNetError {
    fn from(e: EmbeddingError) -> Self {
        bitnet_common::BitNetError::Kernel(KernelError::InvalidArguments { reason: e.to_string() })
    }
}

// ── Configuration ────────────────────────────────────────────────

/// Configuration for an embedding table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingConfig {
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Dimensionality of each embedding vector.
    pub embedding_dim: usize,
    /// Optional padding index — tokens with this ID produce zero vectors.
    pub padding_idx: Option<u32>,
}

impl EmbeddingConfig {
    /// Create a new embedding configuration.
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self { vocab_size, embedding_dim, padding_idx: None }
    }

    /// Set the padding index.
    #[must_use]
    pub fn with_padding_idx(mut self, idx: u32) -> Self {
        self.padding_idx = Some(idx);
        self
    }

    fn weight_len(&self) -> usize {
        self.vocab_size * self.embedding_dim
    }
}

// ── Embedding table ──────────────────────────────────────────────

/// Token embedding table backed by a contiguous `f32` weight matrix.
///
/// Layout: row-major `[vocab_size, embedding_dim]`.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    /// Flat weight buffer.
    pub weight: Vec<f32>,
    /// Configuration.
    pub config: EmbeddingConfig,
}

impl EmbeddingTable {
    /// Create an embedding table, validating the weight length.
    pub fn new(
        weight: Vec<f32>,
        config: EmbeddingConfig,
    ) -> std::result::Result<Self, EmbeddingError> {
        let expected = config.weight_len();
        if weight.len() != expected {
            return Err(EmbeddingError::WeightShapeMismatch { got: weight.len(), expected });
        }
        Ok(Self { weight, config })
    }

    /// Convenience: look up a single token.
    pub fn lookup_one(&self, token_id: u32) -> std::result::Result<Vec<f32>, EmbeddingError> {
        let dim = self.config.embedding_dim;
        let mut out = vec![0.0f32; dim];
        embedding_lookup(&[token_id], &self.weight, &mut out, &self.config)?;
        Ok(out)
    }

    /// Look up embeddings for a batch of token IDs.
    pub fn lookup_batch(
        &self,
        token_ids: &[u32],
        output: &mut [f32],
    ) -> std::result::Result<(), EmbeddingError> {
        embedding_lookup_batch(token_ids, &self.weight, output, &self.config)
    }
}

// ── AVX2 helpers (x86_64 only) ───────────────────────────────────

/// Copy `dim` floats from `src` to `dst` using AVX2 256-bit loads/stores.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_row_avx2(src: &[f32], dst: &mut [f32], dim: usize) {
    use std::arch::x86_64::*;
    let chunks = dim / 8;
    let remainder = dim % 8;
    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(src.as_ptr().add(offset));
            _mm256_storeu_ps(dst.as_mut_ptr().add(offset), v);
        }
    }
    let tail_start = chunks * 8;
    for i in 0..remainder {
        unsafe {
            *dst.get_unchecked_mut(tail_start + i) = *src.get_unchecked(tail_start + i);
        }
    }
}

/// Add `dim` floats from `src` into `dst` in place using AVX2.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_row_avx2(src: &[f32], dst: &mut [f32], dim: usize) {
    use std::arch::x86_64::*;
    let chunks = dim / 8;
    let remainder = dim % 8;
    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let a = _mm256_loadu_ps(dst.as_ptr().add(offset));
            let b = _mm256_loadu_ps(src.as_ptr().add(offset));
            let c = _mm256_add_ps(a, b);
            _mm256_storeu_ps(dst.as_mut_ptr().add(offset), c);
        }
    }
    let tail_start = chunks * 8;
    for i in 0..remainder {
        unsafe {
            *dst.get_unchecked_mut(tail_start + i) += *src.get_unchecked(tail_start + i);
        }
    }
}

/// Multiply every element in `dst[..dim]` by `scalar` using AVX2.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_row_avx2(dst: &mut [f32], dim: usize, scalar: f32) {
    use std::arch::x86_64::*;
    let s = _mm256_set1_ps(scalar);
    let chunks = dim / 8;
    let remainder = dim % 8;
    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let v = _mm256_loadu_ps(dst.as_ptr().add(offset));
            let r = _mm256_mul_ps(v, s);
            _mm256_storeu_ps(dst.as_mut_ptr().add(offset), r);
        }
    }
    let tail_start = chunks * 8;
    for i in 0..remainder {
        unsafe {
            *dst.get_unchecked_mut(tail_start + i) *= scalar;
        }
    }
}

// ── Scalar helpers ───────────────────────────────────────────────

fn copy_row_scalar(src: &[f32], dst: &mut [f32], dim: usize) {
    dst[..dim].copy_from_slice(&src[..dim]);
}

fn add_row_scalar(src: &[f32], dst: &mut [f32], dim: usize) {
    for i in 0..dim {
        dst[i] += src[i];
    }
}

fn scale_row_scalar(dst: &mut [f32], dim: usize, scalar: f32) {
    for v in dst[..dim].iter_mut() {
        *v *= scalar;
    }
}

// ── Runtime dispatch wrappers ────────────────────────────────────

fn copy_row(src: &[f32], dst: &mut [f32], dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature check above guarantees AVX2.
            unsafe { copy_row_avx2(src, dst, dim) };
            return;
        }
    }
    copy_row_scalar(src, dst, dim);
}

fn add_row(src: &[f32], dst: &mut [f32], dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { add_row_avx2(src, dst, dim) };
            return;
        }
    }
    add_row_scalar(src, dst, dim);
}

fn scale_row(dst: &mut [f32], dim: usize, scalar: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { scale_row_avx2(dst, dim, scalar) };
            return;
        }
    }
    scale_row_scalar(dst, dim, scalar);
}

// ── Public API ───────────────────────────────────────────────────

/// Look up the embedding for a single token ID (or a 1-element batch).
///
/// Writes `embedding_dim` floats into `output`.
pub fn embedding_lookup(
    token_ids: &[u32],
    weight: &[f32],
    output: &mut [f32],
    config: &EmbeddingConfig,
) -> std::result::Result<(), EmbeddingError> {
    if token_ids.is_empty() {
        return Ok(());
    }
    let dim = config.embedding_dim;
    let expected_out = token_ids.len() * dim;
    if output.len() < expected_out {
        return Err(EmbeddingError::OutputShapeMismatch {
            got: output.len(),
            expected: expected_out,
        });
    }
    if weight.len() < config.weight_len() {
        return Err(EmbeddingError::WeightShapeMismatch {
            got: weight.len(),
            expected: config.weight_len(),
        });
    }

    for (i, &tid) in token_ids.iter().enumerate() {
        let out_slice = &mut output[i * dim..(i + 1) * dim];
        if config.padding_idx == Some(tid) {
            for v in out_slice.iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        if (tid as usize) >= config.vocab_size {
            return Err(EmbeddingError::TokenOutOfRange {
                token_id: tid,
                vocab_size: config.vocab_size,
            });
        }
        let row_start = (tid as usize) * dim;
        copy_row(&weight[row_start..row_start + dim], out_slice, dim);
    }
    Ok(())
}

/// Batched embedding lookup — semantically identical to [`embedding_lookup`]
/// but named explicitly for batch usage.
pub fn embedding_lookup_batch(
    token_ids: &[u32],
    weight: &[f32],
    output: &mut [f32],
    config: &EmbeddingConfig,
) -> std::result::Result<(), EmbeddingError> {
    embedding_lookup(token_ids, weight, output, config)
}

/// Element-wise sum of embeddings for the given token IDs.
///
/// Output is a single vector of length `embedding_dim`.
pub fn embedding_sum(
    token_ids: &[u32],
    weight: &[f32],
    output: &mut [f32],
    config: &EmbeddingConfig,
) -> std::result::Result<(), EmbeddingError> {
    let dim = config.embedding_dim;
    if output.len() < dim {
        return Err(EmbeddingError::OutputShapeMismatch { got: output.len(), expected: dim });
    }
    if weight.len() < config.weight_len() {
        return Err(EmbeddingError::WeightShapeMismatch {
            got: weight.len(),
            expected: config.weight_len(),
        });
    }
    // Zero the output.
    for v in output[..dim].iter_mut() {
        *v = 0.0;
    }
    for &tid in token_ids {
        if config.padding_idx == Some(tid) {
            continue;
        }
        if (tid as usize) >= config.vocab_size {
            return Err(EmbeddingError::TokenOutOfRange {
                token_id: tid,
                vocab_size: config.vocab_size,
            });
        }
        let row_start = (tid as usize) * dim;
        add_row(&weight[row_start..row_start + dim], &mut output[..dim], dim);
    }
    Ok(())
}

/// Element-wise mean of embeddings for the given token IDs.
///
/// Equivalent to `embedding_sum / n` (skipping padding tokens).
pub fn embedding_mean(
    token_ids: &[u32],
    weight: &[f32],
    output: &mut [f32],
    config: &EmbeddingConfig,
) -> std::result::Result<(), EmbeddingError> {
    embedding_sum(token_ids, weight, output, config)?;
    let dim = config.embedding_dim;
    let count = if let Some(pad) = config.padding_idx {
        token_ids.iter().filter(|&&t| t != pad).count()
    } else {
        token_ids.len()
    };
    if count > 0 {
        let inv = 1.0 / count as f32;
        scale_row(&mut output[..dim], dim, inv);
    }
    Ok(())
}

/// Add absolute position embeddings to an existing embedding buffer.
///
/// `positions` contains the position index for each token in the batch.
/// `pos_weight` is shaped `[max_seq_len, embedding_dim]`.
pub fn position_embedding(
    output: &mut [f32],
    positions: &[usize],
    pos_weight: &[f32],
    embedding_dim: usize,
    max_seq_len: usize,
) -> std::result::Result<(), EmbeddingError> {
    let expected_pos_weight = max_seq_len * embedding_dim;
    if pos_weight.len() < expected_pos_weight {
        return Err(EmbeddingError::WeightShapeMismatch {
            got: pos_weight.len(),
            expected: expected_pos_weight,
        });
    }
    let expected_out = positions.len() * embedding_dim;
    if output.len() < expected_out {
        return Err(EmbeddingError::OutputShapeMismatch {
            got: output.len(),
            expected: expected_out,
        });
    }
    for (i, &pos) in positions.iter().enumerate() {
        if pos >= max_seq_len {
            return Err(EmbeddingError::PositionOutOfRange { position: pos, max_len: max_seq_len });
        }
        let row_start = pos * embedding_dim;
        let out_start = i * embedding_dim;
        add_row(
            &pos_weight[row_start..row_start + embedding_dim],
            &mut output[out_start..out_start + embedding_dim],
            embedding_dim,
        );
    }
    Ok(())
}

/// Apply rotary position embedding (RoPE) in-place.
///
/// Operates on pairs `(x[2k], x[2k+1])` applying the standard rotation:
///
/// ```text
/// x'[2k]   = x[2k]   * cos(θ_k) - x[2k+1] * sin(θ_k)
/// x'[2k+1] = x[2k]   * sin(θ_k) + x[2k+1] * cos(θ_k)
/// ```
///
/// `cos_cache` and `sin_cache` are shaped `[max_seq_len, half_dim]` where
/// `half_dim = embedding_dim / 2`.
pub fn rotary_embedding_apply(
    data: &mut [f32],
    positions: &[usize],
    cos_cache: &[f32],
    sin_cache: &[f32],
    embedding_dim: usize,
    max_seq_len: usize,
) -> std::result::Result<(), EmbeddingError> {
    if !embedding_dim.is_multiple_of(2) {
        return Err(EmbeddingError::OddEmbeddingDim { dim: embedding_dim });
    }
    let half_dim = embedding_dim / 2;
    let expected_cache = max_seq_len * half_dim;
    if cos_cache.len() < expected_cache {
        return Err(EmbeddingError::WeightShapeMismatch {
            got: cos_cache.len(),
            expected: expected_cache,
        });
    }
    if sin_cache.len() < expected_cache {
        return Err(EmbeddingError::WeightShapeMismatch {
            got: sin_cache.len(),
            expected: expected_cache,
        });
    }
    let expected_data = positions.len() * embedding_dim;
    if data.len() < expected_data {
        return Err(EmbeddingError::OutputShapeMismatch {
            got: data.len(),
            expected: expected_data,
        });
    }

    for (i, &pos) in positions.iter().enumerate() {
        if pos >= max_seq_len {
            return Err(EmbeddingError::PositionOutOfRange { position: pos, max_len: max_seq_len });
        }
        let data_offset = i * embedding_dim;
        let cache_offset = pos * half_dim;
        rotary_apply_single(
            &mut data[data_offset..data_offset + embedding_dim],
            &cos_cache[cache_offset..cache_offset + half_dim],
            &sin_cache[cache_offset..cache_offset + half_dim],
            half_dim,
        );
    }
    Ok(())
}

// ── RoPE inner loop with SIMD dispatch ───────────────────────────

fn rotary_apply_single(data: &mut [f32], cos: &[f32], sin: &[f32], half_dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature check above guarantees AVX2.
            unsafe { rotary_apply_avx2(data, cos, sin, half_dim) };
            return;
        }
    }
    rotary_apply_scalar(data, cos, sin, half_dim);
}

fn rotary_apply_scalar(data: &mut [f32], cos: &[f32], sin: &[f32], half_dim: usize) {
    for k in 0..half_dim {
        let x0 = data[2 * k];
        let x1 = data[2 * k + 1];
        let c = cos[k];
        let s = sin[k];
        data[2 * k] = x0 * c - x1 * s;
        data[2 * k + 1] = x0 * s + x1 * c;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rotary_apply_avx2(data: &mut [f32], cos: &[f32], sin: &[f32], half_dim: usize) {
    use std::arch::x86_64::*;

    // Process 4 pairs (8 floats) at a time.
    let chunks = half_dim / 4;
    let remainder = half_dim % 4;

    for i in 0..chunks {
        let cos_off = i * 4;
        let data_off = i * 8;

        unsafe {
            // Load 4 cos/sin values.
            let cv = _mm_loadu_ps(cos.as_ptr().add(cos_off));
            let sv = _mm_loadu_ps(sin.as_ptr().add(cos_off));

            // Deinterleave: evens → x0,x2,x4,x6;  odds → x1,x3,x5,x7
            let evens_lo = _mm_shuffle_ps(
                _mm_loadu_ps(data.as_ptr().add(data_off)),
                _mm_loadu_ps(data.as_ptr().add(data_off + 4)),
                0b10_00_10_00,
            );
            let odds_lo = _mm_shuffle_ps(
                _mm_loadu_ps(data.as_ptr().add(data_off)),
                _mm_loadu_ps(data.as_ptr().add(data_off + 4)),
                0b11_01_11_01,
            );

            // x' = x_even * cos - x_odd * sin
            let r0 = _mm_sub_ps(_mm_mul_ps(evens_lo, cv), _mm_mul_ps(odds_lo, sv));
            // x'' = x_even * sin + x_odd * cos
            let r1 = _mm_add_ps(_mm_mul_ps(evens_lo, sv), _mm_mul_ps(odds_lo, cv));

            // Re-interleave pairs back.
            let lo = _mm_unpacklo_ps(r0, r1);
            let hi = _mm_unpackhi_ps(r0, r1);
            _mm_storeu_ps(data.as_mut_ptr().add(data_off), lo);
            _mm_storeu_ps(data.as_mut_ptr().add(data_off + 4), hi);
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for k in tail_start..tail_start + remainder {
        unsafe {
            let x0 = *data.get_unchecked(2 * k);
            let x1 = *data.get_unchecked(2 * k + 1);
            let c = *cos.get_unchecked(k);
            let s = *sin.get_unchecked(k);
            *data.get_unchecked_mut(2 * k) = x0 * c - x1 * s;
            *data.get_unchecked_mut(2 * k + 1) = x0 * s + x1 * c;
        }
    }
}

// ══════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────

    fn simple_config(vocab: usize, dim: usize) -> EmbeddingConfig {
        EmbeddingConfig::new(vocab, dim)
    }

    /// Build a deterministic weight table where row `i` is filled with `(i+1)` as f32.
    fn sequential_weights(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab).flat_map(|i| std::iter::repeat_n((i + 1) as f32, dim)).collect()
    }

    /// Build weight table where row `i`, col `j` = `i * dim + j` as f32.
    fn indexed_weights(vocab: usize, dim: usize) -> Vec<f32> {
        (0..vocab * dim).map(|i| i as f32).collect()
    }

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol={tol})",);
        }
    }

    // ── EmbeddingConfig tests ───────────────────────────────────

    #[test]
    fn config_new_basic() {
        let c = EmbeddingConfig::new(100, 64);
        assert_eq!(c.vocab_size, 100);
        assert_eq!(c.embedding_dim, 64);
        assert_eq!(c.padding_idx, None);
    }

    #[test]
    fn config_with_padding() {
        let c = EmbeddingConfig::new(100, 64).with_padding_idx(0);
        assert_eq!(c.padding_idx, Some(0));
    }

    #[test]
    fn config_weight_len() {
        let c = EmbeddingConfig::new(50, 32);
        assert_eq!(c.weight_len(), 1600);
    }

    // ── EmbeddingTable tests ────────────────────────────────────

    #[test]
    fn table_new_valid() {
        let w = vec![0.0f32; 4 * 8];
        let t = EmbeddingTable::new(w, simple_config(4, 8));
        assert!(t.is_ok());
    }

    #[test]
    fn table_new_wrong_length() {
        let w = vec![0.0f32; 10];
        let t = EmbeddingTable::new(w, simple_config(4, 8));
        assert!(t.is_err());
        match t.unwrap_err() {
            EmbeddingError::WeightShapeMismatch { got: 10, expected: 32 } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn table_lookup_one() {
        let w = sequential_weights(4, 8);
        let t = EmbeddingTable::new(w, simple_config(4, 8)).unwrap();
        let v = t.lookup_one(2).unwrap();
        assert_eq!(v.len(), 8);
        assert_eq!(v, vec![3.0; 8]);
    }

    #[test]
    fn table_lookup_batch_method() {
        let w = sequential_weights(4, 8);
        let t = EmbeddingTable::new(w, simple_config(4, 8)).unwrap();
        let mut out = vec![0.0f32; 2 * 8];
        t.lookup_batch(&[0, 3], &mut out).unwrap();
        assert_eq!(&out[..8], &[1.0; 8]);
        assert_eq!(&out[8..], &[4.0; 8]);
    }

    // ── embedding_lookup tests ──────────────────────────────────

    #[test]
    fn lookup_empty_ids() {
        let w = vec![0.0f32; 8];
        let mut out = vec![0.0f32; 8];
        embedding_lookup(&[], &w, &mut out, &simple_config(1, 8)).unwrap();
    }

    #[test]
    fn lookup_single_token() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_lookup(&[1], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn lookup_multiple_tokens() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 8];
        embedding_lookup(&[0, 2], &w, &mut out, &cfg).unwrap();
        assert_eq!(&out[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&out[4..], &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn lookup_with_padding_zero_vec() {
        let cfg = simple_config(3, 4).with_padding_idx(1);
        let w = indexed_weights(3, 4);
        let mut out = vec![999.0f32; 4];
        embedding_lookup(&[1], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn lookup_token_out_of_range() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        let err = embedding_lookup(&[5], &w, &mut out, &cfg).unwrap_err();
        assert_eq!(err, EmbeddingError::TokenOutOfRange { token_id: 5, vocab_size: 3 });
    }

    #[test]
    fn lookup_output_too_small() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 2];
        let err = embedding_lookup(&[0], &w, &mut out, &cfg).unwrap_err();
        matches!(err, EmbeddingError::OutputShapeMismatch { .. });
    }

    #[test]
    fn lookup_weight_too_small() {
        let cfg = simple_config(3, 4);
        let w = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        let err = embedding_lookup(&[0], &w, &mut out, &cfg).unwrap_err();
        matches!(err, EmbeddingError::WeightShapeMismatch { .. });
    }

    #[test]
    fn lookup_large_dim_avx_path() {
        // 256 floats exercises the AVX2 loop (256 / 8 = 32 full chunks).
        let dim = 256;
        let cfg = simple_config(2, dim);
        let w: Vec<f32> = (0..2 * dim).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; dim];
        embedding_lookup(&[1], &w, &mut out, &cfg).unwrap();
        let expected: Vec<f32> = (dim..2 * dim).map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn lookup_dim_not_multiple_of_8() {
        let dim = 13;
        let cfg = simple_config(2, dim);
        let w: Vec<f32> = (0..2 * dim).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; dim];
        embedding_lookup(&[1], &w, &mut out, &cfg).unwrap();
        let expected: Vec<f32> = (dim..2 * dim).map(|i| i as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn lookup_dim_1() {
        let cfg = simple_config(5, 1);
        let w = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0f32; 1];
        embedding_lookup(&[3], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![40.0]);
    }

    #[test]
    fn lookup_all_padding() {
        let cfg = simple_config(3, 4).with_padding_idx(0);
        let w = indexed_weights(3, 4);
        let mut out = vec![999.0f32; 12];
        embedding_lookup(&[0, 0, 0], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![0.0; 12]);
    }

    #[test]
    fn lookup_mixed_padding_and_real() {
        let cfg = simple_config(3, 4).with_padding_idx(1);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 8];
        embedding_lookup(&[1, 2], &w, &mut out, &cfg).unwrap();
        assert_eq!(&out[..4], &[0.0; 4]);
        assert_eq!(&out[4..], &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn lookup_first_and_last_token() {
        let cfg = simple_config(4, 2);
        let w = indexed_weights(4, 2);
        let mut out = vec![0.0f32; 4];
        embedding_lookup(&[0, 3], &w, &mut out, &cfg).unwrap();
        assert_eq!(&out[..2], &[0.0, 1.0]);
        assert_eq!(&out[2..], &[6.0, 7.0]);
    }

    #[test]
    fn lookup_duplicate_tokens() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 12];
        embedding_lookup(&[1, 1, 1], &w, &mut out, &cfg).unwrap();
        for chunk in out.chunks(4) {
            assert_eq!(chunk, &[4.0, 5.0, 6.0, 7.0]);
        }
    }

    // ── embedding_lookup_batch tests ────────────────────────────

    #[test]
    fn batch_lookup_same_as_single() {
        let cfg = simple_config(5, 8);
        let w = indexed_weights(5, 8);
        let ids = [0u32, 2, 4];
        let mut out_single = vec![0.0f32; 3 * 8];
        let mut out_batch = vec![0.0f32; 3 * 8];
        embedding_lookup(&ids, &w, &mut out_single, &cfg).unwrap();
        embedding_lookup_batch(&ids, &w, &mut out_batch, &cfg).unwrap();
        assert_eq!(out_single, out_batch);
    }

    // ── embedding_sum tests ─────────────────────────────────────

    #[test]
    fn sum_single_token() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_sum(&[1], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn sum_two_tokens() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_sum(&[0, 2], &w, &mut out, &cfg).unwrap();
        // row0 = [0,1,2,3], row2 = [8,9,10,11] → sum = [8,10,12,14]
        assert_eq!(out, vec![8.0, 10.0, 12.0, 14.0]);
    }

    #[test]
    fn sum_with_padding_skipped() {
        let cfg = simple_config(3, 4).with_padding_idx(0);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_sum(&[0, 2], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn sum_empty() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![999.0f32; 4];
        embedding_sum(&[], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn sum_token_out_of_range() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        let err = embedding_sum(&[10], &w, &mut out, &cfg).unwrap_err();
        assert_eq!(err, EmbeddingError::TokenOutOfRange { token_id: 10, vocab_size: 3 });
    }

    #[test]
    fn sum_output_too_small() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 2];
        let err = embedding_sum(&[0], &w, &mut out, &cfg).unwrap_err();
        matches!(err, EmbeddingError::OutputShapeMismatch { .. });
    }

    #[test]
    fn sum_large_dim() {
        let dim = 256;
        let cfg = simple_config(2, dim);
        let w = sequential_weights(2, dim);
        let mut out = vec![0.0f32; dim];
        embedding_sum(&[0, 1], &w, &mut out, &cfg).unwrap();
        // row0 all 1.0, row1 all 2.0 → sum all 3.0
        assert_eq!(out, vec![3.0; dim]);
    }

    // ── embedding_mean tests ────────────────────────────────────

    #[test]
    fn mean_single_token() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_mean(&[1], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn mean_two_tokens() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_mean(&[0, 2], &w, &mut out, &cfg).unwrap();
        assert_approx_eq(&out, &[4.0, 5.0, 6.0, 7.0], 1e-5);
    }

    #[test]
    fn mean_with_padding() {
        let cfg = simple_config(3, 4).with_padding_idx(0);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_mean(&[0, 2], &w, &mut out, &cfg).unwrap();
        // Only token 2 counts → mean = row2 itself
        assert_eq!(out, vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn mean_all_padding() {
        let cfg = simple_config(3, 4).with_padding_idx(0);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_mean(&[0, 0], &w, &mut out, &cfg).unwrap();
        // count=0 → output stays zero (no division)
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn mean_empty() {
        let cfg = simple_config(3, 4);
        let w = indexed_weights(3, 4);
        let mut out = vec![0.0f32; 4];
        embedding_mean(&[], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![0.0; 4]);
    }

    // ── position_embedding tests ────────────────────────────────

    #[test]
    fn position_embedding_basic() {
        let dim = 4;
        let max_len = 3;
        let pos_w: Vec<f32> = (0..max_len * dim).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![1.0f32; dim];
        position_embedding(&mut out, &[1], &pos_w, dim, max_len).unwrap();
        // row1 = [0.4, 0.5, 0.6, 0.7], added to [1,1,1,1]
        assert_approx_eq(&out, &[1.4, 1.5, 1.6, 1.7], 1e-5);
    }

    #[test]
    fn position_embedding_batch() {
        let dim = 4;
        let max_len = 4;
        let pos_w: Vec<f32> = (0..max_len * dim).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 2 * dim];
        position_embedding(&mut out, &[0, 2], &pos_w, dim, max_len).unwrap();
        assert_eq!(&out[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&out[4..], &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn position_embedding_out_of_range() {
        let dim = 4;
        let max_len = 2;
        let pos_w = vec![0.0f32; max_len * dim];
        let mut out = vec![0.0f32; dim];
        let err = position_embedding(&mut out, &[5], &pos_w, dim, max_len).unwrap_err();
        assert_eq!(err, EmbeddingError::PositionOutOfRange { position: 5, max_len: 2 });
    }

    #[test]
    fn position_embedding_weight_too_small() {
        let dim = 4;
        let max_len = 2;
        let pos_w = vec![0.0f32; 2]; // too small
        let mut out = vec![0.0f32; dim];
        let err = position_embedding(&mut out, &[0], &pos_w, dim, max_len).unwrap_err();
        matches!(err, EmbeddingError::WeightShapeMismatch { .. });
    }

    #[test]
    fn position_embedding_output_too_small() {
        let dim = 4;
        let max_len = 2;
        let pos_w = vec![0.0f32; max_len * dim];
        let mut out = vec![0.0f32; 2]; // too small
        let err = position_embedding(&mut out, &[0], &pos_w, dim, max_len).unwrap_err();
        matches!(err, EmbeddingError::OutputShapeMismatch { .. });
    }

    #[test]
    fn position_embedding_additive() {
        let dim = 4;
        let max_len = 2;
        let pos_w = vec![10.0; max_len * dim];
        let mut out = vec![5.0f32; dim];
        position_embedding(&mut out, &[0], &pos_w, dim, max_len).unwrap();
        assert_eq!(out, vec![15.0; dim]);
    }

    // ── rotary_embedding_apply tests ────────────────────────────

    fn build_rope_caches(max_len: usize, half_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
        let mut cos_cache = vec![0.0f32; max_len * half_dim];
        let mut sin_cache = vec![0.0f32; max_len * half_dim];
        for pos in 0..max_len {
            for k in 0..half_dim {
                let freq = 1.0 / base.powf(2.0 * k as f32 / (2.0 * half_dim as f32));
                let angle = pos as f32 * freq;
                cos_cache[pos * half_dim + k] = angle.cos();
                sin_cache[pos * half_dim + k] = angle.sin();
            }
        }
        (cos_cache, sin_cache)
    }

    #[test]
    fn rope_identity_at_pos_zero() {
        let dim = 8;
        let half = dim / 2;
        let max_len = 4;
        let (cos_c, sin_c) = build_rope_caches(max_len, half, 10000.0);
        // At position 0 all angles are 0 → cos=1, sin=0 → data unchanged.
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = data.clone();
        rotary_embedding_apply(&mut data, &[0], &cos_c, &sin_c, dim, max_len).unwrap();
        assert_approx_eq(&data, &original, 1e-5);
    }

    #[test]
    fn rope_odd_dim_rejected() {
        let err =
            rotary_embedding_apply(&mut [0.0; 3], &[0], &[0.0; 4], &[0.0; 4], 3, 4).unwrap_err();
        assert_eq!(err, EmbeddingError::OddEmbeddingDim { dim: 3 });
    }

    #[test]
    fn rope_position_out_of_range() {
        let dim = 4;
        let half = 2;
        let max_len = 2;
        let cos_c = vec![1.0; max_len * half];
        let sin_c = vec![0.0; max_len * half];
        let mut data = vec![1.0; dim];
        let err =
            rotary_embedding_apply(&mut data, &[5], &cos_c, &sin_c, dim, max_len).unwrap_err();
        assert_eq!(err, EmbeddingError::PositionOutOfRange { position: 5, max_len: 2 });
    }

    #[test]
    fn rope_known_rotation() {
        // Manual rotation: cos=0, sin=1 → x'[0] = -x[1], x'[1] = x[0]
        let dim = 2;
        let _half = 1;
        let max_len = 1;
        let cos_c = vec![0.0];
        let sin_c = vec![1.0];
        let mut data = vec![3.0, 4.0];
        rotary_embedding_apply(&mut data, &[0], &cos_c, &sin_c, dim, max_len).unwrap();
        assert_approx_eq(&data, &[-4.0, 3.0], 1e-5);
    }

    #[test]
    fn rope_batch_positions() {
        let dim = 4;
        let half = 2;
        let max_len = 4;
        let (cos_c, sin_c) = build_rope_caches(max_len, half, 10000.0);
        let mut data = vec![1.0f32; 2 * dim];
        rotary_embedding_apply(&mut data, &[0, 2], &cos_c, &sin_c, dim, max_len).unwrap();
        // Position 0 → identity, position 2 → some rotation applied.
        // Just verify position 0 is unchanged.
        assert_approx_eq(&data[..dim], &[1.0; 4], 1e-5);
    }

    #[test]
    fn rope_cos_cache_too_small() {
        let dim = 4;
        let max_len = 2;
        let cos_c = vec![0.0; 1]; // too small
        let sin_c = vec![0.0; max_len * 2];
        let mut data = vec![0.0; dim];
        let err =
            rotary_embedding_apply(&mut data, &[0], &cos_c, &sin_c, dim, max_len).unwrap_err();
        matches!(err, EmbeddingError::WeightShapeMismatch { .. });
    }

    #[test]
    fn rope_sin_cache_too_small() {
        let dim = 4;
        let max_len = 2;
        let cos_c = vec![0.0; max_len * 2];
        let sin_c = vec![0.0; 1]; // too small
        let mut data = vec![0.0; dim];
        let err =
            rotary_embedding_apply(&mut data, &[0], &cos_c, &sin_c, dim, max_len).unwrap_err();
        matches!(err, EmbeddingError::WeightShapeMismatch { .. });
    }

    #[test]
    fn rope_data_too_small() {
        let dim = 4;
        let max_len = 2;
        let cos_c = vec![0.0; max_len * 2];
        let sin_c = vec![0.0; max_len * 2];
        let mut data = vec![0.0; 2]; // too small for dim=4
        let err =
            rotary_embedding_apply(&mut data, &[0], &cos_c, &sin_c, dim, max_len).unwrap_err();
        matches!(err, EmbeddingError::OutputShapeMismatch { .. });
    }

    #[test]
    fn rope_large_dim_exercises_avx2() {
        let dim = 256;
        let half = dim / 2;
        let max_len = 2;
        let (cos_c, sin_c) = build_rope_caches(max_len, half, 10000.0);
        let mut data: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let original = data.clone();
        // Position 0 → identity.
        rotary_embedding_apply(&mut data, &[0], &cos_c, &sin_c, dim, max_len).unwrap();
        assert_approx_eq(&data, &original, 1e-4);
    }

    // ── EmbeddingError Display tests ────────────────────────────

    #[test]
    fn error_display_token_out_of_range() {
        let e = EmbeddingError::TokenOutOfRange { token_id: 99, vocab_size: 50 };
        assert!(e.to_string().contains("99"));
        assert!(e.to_string().contains("50"));
    }

    #[test]
    fn error_display_weight_mismatch() {
        let e = EmbeddingError::WeightShapeMismatch { got: 10, expected: 20 };
        assert!(e.to_string().contains("10"));
        assert!(e.to_string().contains("20"));
    }

    #[test]
    fn error_display_output_mismatch() {
        let e = EmbeddingError::OutputShapeMismatch { got: 5, expected: 10 };
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn error_display_position_out_of_range() {
        let e = EmbeddingError::PositionOutOfRange { position: 100, max_len: 50 };
        assert!(e.to_string().contains("100"));
    }

    #[test]
    fn error_display_odd_dim() {
        let e = EmbeddingError::OddEmbeddingDim { dim: 7 };
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn error_display_invalid_argument() {
        let e = EmbeddingError::InvalidArgument("bad input".into());
        assert!(e.to_string().contains("bad input"));
    }

    #[test]
    fn error_into_bitnet_error() {
        let e = EmbeddingError::TokenOutOfRange { token_id: 1, vocab_size: 1 };
        let be: bitnet_common::BitNetError = e.into();
        let msg = format!("{be}");
        assert!(msg.contains("token id 1"));
    }

    // ── Scalar vs dispatch parity tests ─────────────────────────

    #[test]
    fn copy_row_scalar_parity() {
        let src: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let mut dst_dispatch = vec![0.0f32; 33];
        let mut dst_scalar = vec![0.0f32; 33];
        copy_row(&src, &mut dst_dispatch, 33);
        copy_row_scalar(&src, &mut dst_scalar, 33);
        assert_eq!(dst_dispatch, dst_scalar);
    }

    #[test]
    fn add_row_scalar_parity() {
        let src: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let mut dst_dispatch = vec![1.0f32; 33];
        let mut dst_scalar = vec![1.0f32; 33];
        add_row(&src, &mut dst_dispatch, 33);
        add_row_scalar(&src, &mut dst_scalar, 33);
        assert_eq!(dst_dispatch, dst_scalar);
    }

    #[test]
    fn scale_row_scalar_parity() {
        let mut dst_dispatch: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let mut dst_scalar = dst_dispatch.clone();
        scale_row(&mut dst_dispatch, 33, 0.5);
        scale_row_scalar(&mut dst_scalar, 33, 0.5);
        assert_approx_eq(&dst_dispatch, &dst_scalar, 1e-6);
    }

    // ── Additional edge-case tests ──────────────────────────────

    #[test]
    fn lookup_vocab_size_1() {
        let cfg = simple_config(1, 4);
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        embedding_lookup(&[0], &w, &mut out, &cfg).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sum_three_tokens() {
        let cfg = simple_config(3, 2);
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0f32; 2];
        embedding_sum(&[0, 1, 2], &w, &mut out, &cfg).unwrap();
        assert_approx_eq(&out, &[9.0, 12.0], 1e-5);
    }

    #[test]
    fn mean_three_tokens() {
        let cfg = simple_config(3, 2);
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0f32; 2];
        embedding_mean(&[0, 1, 2], &w, &mut out, &cfg).unwrap();
        assert_approx_eq(&out, &[3.0, 4.0], 1e-5);
    }

    #[test]
    fn position_embedding_empty_positions() {
        let dim = 4;
        let max_len = 2;
        let pos_w = vec![0.0f32; max_len * dim];
        let mut out = vec![1.0f32; 0];
        position_embedding(&mut out, &[], &pos_w, dim, max_len).unwrap();
    }

    #[test]
    fn rope_empty_positions() {
        let dim = 4;
        let max_len = 2;
        let cos_c = vec![0.0; max_len * 2];
        let sin_c = vec![0.0; max_len * 2];
        let mut data = vec![0.0f32; 0];
        rotary_embedding_apply(&mut data, &[], &cos_c, &sin_c, dim, max_len).unwrap();
    }

    #[test]
    fn rope_double_apply_and_reverse() {
        // Applying RoPE at pos p then at pos -p (via negated sin) should restore original.
        let dim = 4;
        let half = 2;
        let max_len = 2;
        let (cos_c, sin_c) = build_rope_caches(max_len, half, 10000.0);
        let neg_sin: Vec<f32> = sin_c.iter().map(|s| -s).collect();
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        rotary_embedding_apply(&mut data, &[1], &cos_c, &sin_c, dim, max_len).unwrap();
        rotary_embedding_apply(&mut data, &[1], &cos_c, &neg_sin, dim, max_len).unwrap();
        assert_approx_eq(&data, &original, 1e-4);
    }

    // ── proptest properties ─────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn embedding_strategy(
            max_vocab: usize,
            max_dim: usize,
        ) -> impl Strategy<Value = (EmbeddingConfig, Vec<f32>)> {
            (1..=max_vocab, 1..=max_dim).prop_flat_map(|(v, d)| {
                let len = v * d;
                (Just(simple_config(v, d)), proptest::collection::vec(-10.0f32..10.0, len..=len))
            })
        }

        proptest! {
            #![proptest_config(proptest::prelude::ProptestConfig::with_cases(256))]

            #[test]
            fn prop_lookup_output_matches_weight_row(
                (cfg, w) in embedding_strategy(32, 64),
            ) {
                let vocab = cfg.vocab_size;
                let dim = cfg.embedding_dim;
                for tid in 0..vocab {
                    let mut out = vec![0.0f32; dim];
                    embedding_lookup(&[tid as u32], &w, &mut out, &cfg).unwrap();
                    let row = &w[tid * dim..(tid + 1) * dim];
                    prop_assert_eq!(&out, &row.to_vec());
                }
            }

            #[test]
            fn prop_sum_is_additive(
                (cfg, w) in embedding_strategy(16, 32),
            ) {
                let vocab = cfg.vocab_size;
                let dim = cfg.embedding_dim;
                if vocab < 2 { return Ok(()); }
                let ids = vec![0u32, 1];
                let mut sum_out = vec![0.0f32; dim];
                embedding_sum(&ids, &w, &mut sum_out, &cfg).unwrap();

                let mut e0 = vec![0.0f32; dim];
                let mut e1 = vec![0.0f32; dim];
                embedding_lookup(&[0], &w, &mut e0, &cfg).unwrap();
                embedding_lookup(&[1], &w, &mut e1, &cfg).unwrap();
                let expected: Vec<f32> = e0.iter().zip(e1.iter()).map(|(a, b)| a + b).collect();
                for (i, (a, b)) in sum_out.iter().zip(expected.iter()).enumerate() {
                    prop_assert!((a - b).abs() < 1e-4, "mismatch at {i}: {a} vs {b}");
                }
            }

            #[test]
            fn prop_mean_bounded_by_extremes(
                (cfg, w) in embedding_strategy(16, 32),
            ) {
                let vocab = cfg.vocab_size;
                let dim = cfg.embedding_dim;
                if vocab == 0 { return Ok(()); }
                let ids: Vec<u32> = (0..vocab as u32).collect();
                let mut mean_out = vec![0.0f32; dim];
                embedding_mean(&ids, &w, &mut mean_out, &cfg).unwrap();

                for j in 0..dim {
                    let col_min = (0..vocab).map(|i| w[i * dim + j]).fold(f32::INFINITY, f32::min);
                    let col_max = (0..vocab).map(|i| w[i * dim + j]).fold(f32::NEG_INFINITY, f32::max);
                    prop_assert!(
                        mean_out[j] >= col_min - 1e-4 && mean_out[j] <= col_max + 1e-4,
                        "mean[{j}] = {} not in [{col_min}, {col_max}]", mean_out[j],
                    );
                }
            }

            #[test]
            fn prop_padding_produces_zeros(
                (cfg_base, w) in embedding_strategy(16, 32),
            ) {
                let cfg = EmbeddingConfig {
                    padding_idx: Some(0),
                    ..cfg_base
                };
                let dim = cfg.embedding_dim;
                let mut out = vec![999.0f32; dim];
                embedding_lookup(&[0], &w, &mut out, &cfg).unwrap();
                for (idx, v) in out.iter().enumerate() {
                    prop_assert_eq!(*v, 0.0, "expected zero at index {}", idx);
                }
            }

            #[test]
            fn prop_rope_preserves_norm(
                half_dim in 1usize..33,
                pos in 0usize..8,
            ) {
                let dim = half_dim * 2;
                let max_len = 8;
                let (cos_c, sin_c) = build_rope_caches(max_len, half_dim, 10000.0);
                let data_orig: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect();
                let norm_before: f32 = data_orig.iter().map(|x| x * x).sum::<f32>().sqrt();

                let mut data = data_orig;
                rotary_embedding_apply(&mut data, &[pos], &cos_c, &sin_c, dim, max_len)
                    .unwrap();
                let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
                prop_assert!(
                    (norm_before - norm_after).abs() < 1e-2,
                    "norm changed: {norm_before} → {norm_after}",
                );
            }
        }
    }
}
