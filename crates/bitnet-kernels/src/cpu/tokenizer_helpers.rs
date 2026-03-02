//! CPU helper kernels for tokenizer-related tensor operations.
//!
//! These functions bridge the gap between tokenizer output (token IDs,
//! padding information) and the tensor representations consumed by
//! transformer layers: one-hot encodings, embedding lookups, causal and
//! padding masks, position IDs, and sequence padding/truncation.
//!
//! All operations work on flat `f32` / `u32` buffers with mandatory
//! bounds checking and return `Result` on invalid inputs.

use bitnet_common::{BitNetError, KernelError, Result};
use std::fmt;

// ── Error type ─────────────────────────────────────────────────────

/// Errors specific to tokenizer helper operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerHelperError {
    /// A required argument was invalid (empty, zero-length, out-of-range).
    InvalidArgument(String),
    /// A token ID exceeded the vocabulary size.
    VocabOutOfBounds { token_id: u32, vocab_size: u32 },
    /// A sequence was shorter or longer than expected.
    SequenceLengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for TokenizerHelperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(reason) => write!(f, "tokenizer helper: {reason}"),
            Self::VocabOutOfBounds { token_id, vocab_size } => {
                write!(
                    f,
                    "tokenizer helper: token ID {token_id} out of bounds for vocab size {vocab_size}"
                )
            }
            Self::SequenceLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "tokenizer helper: sequence length mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for TokenizerHelperError {}

// ── Strategy enums ─────────────────────────────────────────────────

/// Where to add padding tokens when a sequence is shorter than the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad on the left (prepend padding tokens).
    Left,
    /// Pad on the right (append padding tokens).
    Right,
}

/// Which end to remove tokens from when a sequence is too long.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Keep the rightmost tokens (remove from the left).
    Left,
    /// Keep the leftmost tokens (remove from the right).
    Right,
    /// Keep tokens from the center, removing equally from both ends.
    Center,
}

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_args(reason: impl Into<String>) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.into() })
}

fn vocab_oob(token_id: u32, vocab_size: u32) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments {
        reason: format!("token ID {token_id} out of bounds for vocab size {vocab_size}"),
    })
}

// ── One-hot encoding ───────────────────────────────────────────────

/// Convert a slice of token IDs into a flat one-hot matrix.
///
/// Returns a row-major `f32` buffer of shape `[token_ids.len(), vocab_size]`
/// where each row has a single `1.0` at the column matching the token ID.
///
/// # Errors
///
/// Returns an error if `vocab_size` is zero or any token ID ≥ `vocab_size`.
pub fn one_hot_encode(token_ids: &[u32], vocab_size: u32) -> Result<Vec<f32>> {
    if vocab_size == 0 {
        return Err(invalid_args("one_hot_encode: vocab_size must be > 0"));
    }
    let vs = vocab_size as usize;
    let mut out = vec![0.0f32; token_ids.len() * vs];
    for (i, &tid) in token_ids.iter().enumerate() {
        if tid >= vocab_size {
            return Err(vocab_oob(tid, vocab_size));
        }
        out[i * vs + tid as usize] = 1.0;
    }
    Ok(out)
}

// ── Embedding lookup ───────────────────────────────────────────────

/// Batch lookup: map token IDs to embedding vectors.
///
/// `embedding_table` is a flat row-major buffer of shape
/// `[vocab_size, embed_dim]`.  For each token ID the corresponding row
/// is copied into the output.
///
/// Returns a `Vec<f32>` of length `token_ids.len() * embed_dim`.
///
/// # Errors
///
/// Returns an error if dimensions are zero, the table is too short,
/// or any token ID is out of bounds.
pub fn token_ids_to_embeddings(
    embedding_table: &[f32],
    vocab_size: u32,
    embed_dim: usize,
    token_ids: &[u32],
) -> Result<Vec<f32>> {
    if vocab_size == 0 || embed_dim == 0 {
        return Err(invalid_args("token_ids_to_embeddings: vocab_size and embed_dim must be > 0"));
    }
    let vs = vocab_size as usize;
    if embedding_table.len() < vs * embed_dim {
        return Err(invalid_args(format!(
            "token_ids_to_embeddings: table length {} < expected {}",
            embedding_table.len(),
            vs * embed_dim,
        )));
    }
    let mut out = Vec::with_capacity(token_ids.len() * embed_dim);
    for &tid in token_ids {
        if tid >= vocab_size {
            return Err(vocab_oob(tid, vocab_size));
        }
        let start = tid as usize * embed_dim;
        out.extend_from_slice(&embedding_table[start..start + embed_dim]);
    }
    Ok(out)
}

// ── Mask creation ──────────────────────────────────────────────────

/// Create a lower-triangular causal (autoregressive) attention mask.
///
/// Returns a flat row-major `f32` buffer of shape `[seq_len, seq_len]`
/// where `mask[i][j] = 1.0` if `j <= i`, else `0.0`.
///
/// # Errors
///
/// Returns an error if `seq_len` is zero.
pub fn create_causal_mask(seq_len: usize) -> Result<Vec<f32>> {
    if seq_len == 0 {
        return Err(invalid_args("create_causal_mask: seq_len must be > 0"));
    }
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = 1.0;
        }
    }
    Ok(mask)
}

/// Create a padding mask from a batch of sequences.
///
/// `token_ids` is a flat row-major buffer of shape `[batch_size, seq_len]`.
/// The returned mask has the same shape: `1.0` for real tokens, `0.0` for
/// positions equal to `pad_token_id`.
///
/// # Errors
///
/// Returns an error if dimensions are zero or the buffer length does not
/// match `batch_size * seq_len`.
pub fn create_padding_mask(
    token_ids: &[u32],
    batch_size: usize,
    seq_len: usize,
    pad_token_id: u32,
) -> Result<Vec<f32>> {
    if batch_size == 0 || seq_len == 0 {
        return Err(invalid_args("create_padding_mask: batch_size and seq_len must be > 0"));
    }
    let expected = batch_size * seq_len;
    if token_ids.len() != expected {
        return Err(invalid_args(format!(
            "create_padding_mask: token_ids length {} != expected {}",
            token_ids.len(),
            expected,
        )));
    }
    let mask: Vec<f32> =
        token_ids.iter().map(|&t| if t == pad_token_id { 0.0 } else { 1.0 }).collect();
    Ok(mask)
}

/// Combine a causal mask with a padding mask via element-wise AND.
///
/// Both masks must be flat buffers of the same length.  The result is
/// `1.0` only where **both** inputs are `1.0`.
///
/// # Errors
///
/// Returns an error if the lengths differ or either is empty.
pub fn combine_masks(causal: &[f32], padding: &[f32]) -> Result<Vec<f32>> {
    if causal.is_empty() || padding.is_empty() {
        return Err(invalid_args("combine_masks: masks must not be empty"));
    }
    if causal.len() != padding.len() {
        return Err(invalid_args(format!(
            "combine_masks: length mismatch: causal={} padding={}",
            causal.len(),
            padding.len(),
        )));
    }
    let combined: Vec<f32> = causal
        .iter()
        .zip(padding.iter())
        .map(|(&c, &p)| if c > 0.0 && p > 0.0 { 1.0 } else { 0.0 })
        .collect();
    Ok(combined)
}

// ── Padding ────────────────────────────────────────────────────────

/// Left-pad a token sequence to `target_len` with `pad_token_id`.
///
/// If the sequence is already at least `target_len`, it is returned
/// unchanged (as a new `Vec`).
///
/// # Errors
///
/// Returns an error if `target_len` is zero.
pub fn left_pad_sequence(
    token_ids: &[u32],
    target_len: usize,
    pad_token_id: u32,
) -> Result<Vec<u32>> {
    if target_len == 0 {
        return Err(invalid_args("left_pad_sequence: target_len must be > 0"));
    }
    if token_ids.len() >= target_len {
        return Ok(token_ids.to_vec());
    }
    let pad_count = target_len - token_ids.len();
    let mut out = vec![pad_token_id; pad_count];
    out.extend_from_slice(token_ids);
    Ok(out)
}

/// Right-pad a token sequence to `target_len` with `pad_token_id`.
///
/// If the sequence is already at least `target_len`, it is returned
/// unchanged (as a new `Vec`).
///
/// # Errors
///
/// Returns an error if `target_len` is zero.
pub fn right_pad_sequence(
    token_ids: &[u32],
    target_len: usize,
    pad_token_id: u32,
) -> Result<Vec<u32>> {
    if target_len == 0 {
        return Err(invalid_args("right_pad_sequence: target_len must be > 0"));
    }
    if token_ids.len() >= target_len {
        return Ok(token_ids.to_vec());
    }
    let mut out = Vec::with_capacity(target_len);
    out.extend_from_slice(token_ids);
    out.resize(target_len, pad_token_id);
    Ok(out)
}

// ── Truncation ─────────────────────────────────────────────────────

/// Truncate a token sequence to at most `max_len` tokens.
///
/// The `strategy` controls which tokens are kept:
/// - [`TruncationStrategy::Right`]: keep the first `max_len` tokens.
/// - [`TruncationStrategy::Left`]: keep the last `max_len` tokens.
/// - [`TruncationStrategy::Center`]: keep tokens from the middle,
///   removing roughly equal amounts from each end.
///
/// If the sequence is already at most `max_len`, it is returned unchanged.
///
/// # Errors
///
/// Returns an error if `max_len` is zero.
pub fn truncate_sequence(
    token_ids: &[u32],
    max_len: usize,
    strategy: TruncationStrategy,
) -> Result<Vec<u32>> {
    if max_len == 0 {
        return Err(invalid_args("truncate_sequence: max_len must be > 0"));
    }
    if token_ids.len() <= max_len {
        return Ok(token_ids.to_vec());
    }
    let out = match strategy {
        TruncationStrategy::Right => token_ids[..max_len].to_vec(),
        TruncationStrategy::Left => token_ids[token_ids.len() - max_len..].to_vec(),
        TruncationStrategy::Center => {
            let remove = token_ids.len() - max_len;
            let remove_left = remove / 2;
            token_ids[remove_left..remove_left + max_len].to_vec()
        }
    };
    Ok(out)
}

// ── Position IDs ───────────────────────────────────────────────────

/// Generate position IDs for a single sequence, accounting for padding.
///
/// Non-padding positions receive incrementing IDs starting from 0.
/// Padding positions receive `0` (they will be masked out).
///
/// `token_ids` is the raw sequence; `pad_token_id` identifies padding.
///
/// # Errors
///
/// Returns an error if `token_ids` is empty.
pub fn create_position_ids(token_ids: &[u32], pad_token_id: u32) -> Result<Vec<u32>> {
    if token_ids.is_empty() {
        return Err(invalid_args("create_position_ids: token_ids must not be empty"));
    }
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut pos: u32 = 0;
    for &tid in token_ids {
        if tid == pad_token_id {
            positions.push(0);
        } else {
            positions.push(pos);
            pos += 1;
        }
    }
    Ok(positions)
}

/// Batch version of [`create_position_ids`].
///
/// `token_ids` is a flat row-major buffer of shape `[batch_size, seq_len]`.
/// Returns a flat buffer of the same shape with position IDs per sequence.
///
/// # Errors
///
/// Returns an error if dimensions are zero or the buffer length does not
/// match `batch_size * seq_len`.
pub fn batch_encode_positions(
    token_ids: &[u32],
    batch_size: usize,
    seq_len: usize,
    pad_token_id: u32,
) -> Result<Vec<u32>> {
    if batch_size == 0 || seq_len == 0 {
        return Err(invalid_args("batch_encode_positions: batch_size and seq_len must be > 0"));
    }
    let expected = batch_size * seq_len;
    if token_ids.len() != expected {
        return Err(invalid_args(format!(
            "batch_encode_positions: token_ids length {} != expected {}",
            token_ids.len(),
            expected,
        )));
    }
    let mut out = Vec::with_capacity(expected);
    for b in 0..batch_size {
        let seq = &token_ids[b * seq_len..(b + 1) * seq_len];
        // Unwrap safe: seq_len > 0 guaranteed above.
        let positions = create_position_ids(seq, pad_token_id)?;
        out.extend_from_slice(&positions);
    }
    Ok(out)
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── one_hot_encode ─────────────────────────────────────────────

    #[test]
    fn one_hot_basic() {
        let oh = one_hot_encode(&[0, 2], 4).unwrap();
        // row 0: [1,0,0,0], row 1: [0,0,1,0]
        assert_eq!(oh.len(), 8);
        assert_eq!(oh[0], 1.0);
        assert_eq!(oh[1..4], [0.0, 0.0, 0.0]);
        assert_eq!(oh[4..6], [0.0, 0.0]);
        assert_eq!(oh[6], 1.0);
        assert_eq!(oh[7], 0.0);
    }

    #[test]
    fn one_hot_single_token() {
        let oh = one_hot_encode(&[1], 3).unwrap();
        assert_eq!(oh, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn one_hot_empty_tokens() {
        let oh = one_hot_encode(&[], 5).unwrap();
        assert!(oh.is_empty());
    }

    #[test]
    fn one_hot_zero_vocab() {
        let err = one_hot_encode(&[0], 0).unwrap_err();
        assert!(err.to_string().contains("vocab_size must be > 0"));
    }

    #[test]
    fn one_hot_oob_token() {
        let err = one_hot_encode(&[5], 4).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn one_hot_row_sums_to_one() {
        let oh = one_hot_encode(&[0, 1, 2, 3], 4).unwrap();
        for row in 0..4 {
            let sum: f32 = oh[row * 4..(row + 1) * 4].iter().sum();
            assert!((sum - 1.0).abs() < f32::EPSILON);
        }
    }

    // ── token_ids_to_embeddings ────────────────────────────────────

    #[test]
    fn embedding_lookup_basic() {
        let table = [
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
        ];
        let out = token_ids_to_embeddings(&table, 3, 2, &[2, 0]).unwrap();
        assert_eq!(out, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn embedding_lookup_single() {
        let table = [10.0, 20.0, 30.0];
        let out = token_ids_to_embeddings(&table, 1, 3, &[0]).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn embedding_lookup_empty_ids() {
        let table = [1.0, 2.0];
        let out = token_ids_to_embeddings(&table, 1, 2, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn embedding_lookup_oob() {
        let table = [1.0, 2.0, 3.0, 4.0];
        let err = token_ids_to_embeddings(&table, 2, 2, &[5]).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn embedding_lookup_zero_vocab() {
        let err = token_ids_to_embeddings(&[], 0, 4, &[]).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    #[test]
    fn embedding_lookup_zero_dim() {
        let err = token_ids_to_embeddings(&[], 4, 0, &[]).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    #[test]
    fn embedding_lookup_table_too_short() {
        let table = [1.0, 2.0];
        let err = token_ids_to_embeddings(&table, 2, 2, &[0]).unwrap_err();
        assert!(err.to_string().contains("table length"));
    }

    #[test]
    fn embedding_duplicate_ids() {
        let table = [1.0, 2.0, 3.0, 4.0]; // 2×2
        let out = token_ids_to_embeddings(&table, 2, 2, &[1, 1, 0]).unwrap();
        assert_eq!(out, vec![3.0, 4.0, 3.0, 4.0, 1.0, 2.0]);
    }

    // ── create_causal_mask ─────────────────────────────────────────

    #[test]
    fn causal_mask_3x3() {
        let mask = create_causal_mask(3).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn causal_mask_1x1() {
        let mask = create_causal_mask(1).unwrap();
        assert_eq!(mask, vec![1.0]);
    }

    #[test]
    fn causal_mask_zero() {
        let err = create_causal_mask(0).unwrap_err();
        assert!(err.to_string().contains("seq_len must be > 0"));
    }

    #[test]
    fn causal_mask_is_lower_triangular() {
        let n = 5;
        let mask = create_causal_mask(n).unwrap();
        for i in 0..n {
            for j in 0..n {
                let val = mask[i * n + j];
                if j <= i {
                    assert_eq!(val, 1.0, "expected 1.0 at ({i},{j})");
                } else {
                    assert_eq!(val, 0.0, "expected 0.0 at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn causal_mask_diagonal_all_ones() {
        let n = 4;
        let mask = create_causal_mask(n).unwrap();
        for i in 0..n {
            assert_eq!(mask[i * n + i], 1.0);
        }
    }

    // ── create_padding_mask ────────────────────────────────────────

    #[test]
    fn padding_mask_basic() {
        let ids = [0, 1, 2, 0]; // pad=0
        let mask = create_padding_mask(&ids, 1, 4, 0).unwrap();
        assert_eq!(mask, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn padding_mask_no_padding() {
        let ids = [1, 2, 3];
        let mask = create_padding_mask(&ids, 1, 3, 0).unwrap();
        assert_eq!(mask, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn padding_mask_all_padding() {
        let ids = [0, 0, 0];
        let mask = create_padding_mask(&ids, 1, 3, 0).unwrap();
        assert_eq!(mask, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn padding_mask_batch() {
        let ids = [0, 1, 2, 3, 0, 4]; // batch=2, seq=3, pad=0
        let mask = create_padding_mask(&ids, 2, 3, 0).unwrap();
        assert_eq!(mask, vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn padding_mask_length_mismatch() {
        let ids = [1, 2, 3];
        let err = create_padding_mask(&ids, 2, 3, 0).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn padding_mask_zero_batch() {
        let err = create_padding_mask(&[], 0, 3, 0).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    // ── combine_masks ──────────────────────────────────────────────

    #[test]
    fn combine_masks_basic() {
        let causal = [1.0, 0.0, 1.0, 1.0];
        let padding = [1.0, 1.0, 0.0, 1.0];
        let combined = combine_masks(&causal, &padding).unwrap();
        assert_eq!(combined, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn combine_masks_all_ones() {
        let a = [1.0; 4];
        let b = [1.0; 4];
        let c = combine_masks(&a, &b).unwrap();
        assert_eq!(c, vec![1.0; 4]);
    }

    #[test]
    fn combine_masks_all_zeros() {
        let a = [0.0; 4];
        let b = [1.0; 4];
        let c = combine_masks(&a, &b).unwrap();
        assert_eq!(c, vec![0.0; 4]);
    }

    #[test]
    fn combine_masks_length_mismatch() {
        let err = combine_masks(&[1.0, 0.0], &[1.0]).unwrap_err();
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn combine_masks_empty() {
        let err = combine_masks(&[], &[]).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    // ── left_pad_sequence ──────────────────────────────────────────

    #[test]
    fn left_pad_basic() {
        let seq = [1, 2, 3];
        let padded = left_pad_sequence(&seq, 5, 0).unwrap();
        assert_eq!(padded, vec![0, 0, 1, 2, 3]);
    }

    #[test]
    fn left_pad_no_padding_needed() {
        let seq = [1, 2, 3];
        let padded = left_pad_sequence(&seq, 3, 0).unwrap();
        assert_eq!(padded, vec![1, 2, 3]);
    }

    #[test]
    fn left_pad_longer_than_target() {
        let seq = [1, 2, 3, 4, 5];
        let padded = left_pad_sequence(&seq, 3, 0).unwrap();
        assert_eq!(padded, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn left_pad_empty_sequence() {
        let padded = left_pad_sequence(&[], 3, 99).unwrap();
        assert_eq!(padded, vec![99, 99, 99]);
    }

    #[test]
    fn left_pad_zero_target() {
        let err = left_pad_sequence(&[1], 0, 0).unwrap_err();
        assert!(err.to_string().contains("target_len must be > 0"));
    }

    // ── right_pad_sequence ─────────────────────────────────────────

    #[test]
    fn right_pad_basic() {
        let seq = [1, 2, 3];
        let padded = right_pad_sequence(&seq, 5, 0).unwrap();
        assert_eq!(padded, vec![1, 2, 3, 0, 0]);
    }

    #[test]
    fn right_pad_no_padding_needed() {
        let seq = [1, 2, 3];
        let padded = right_pad_sequence(&seq, 2, 0).unwrap();
        assert_eq!(padded, vec![1, 2, 3]);
    }

    #[test]
    fn right_pad_empty_sequence() {
        let padded = right_pad_sequence(&[], 2, 42).unwrap();
        assert_eq!(padded, vec![42, 42]);
    }

    #[test]
    fn right_pad_zero_target() {
        let err = right_pad_sequence(&[1], 0, 0).unwrap_err();
        assert!(err.to_string().contains("target_len must be > 0"));
    }

    // ── truncate_sequence ──────────────────────────────────────────

    #[test]
    fn truncate_right() {
        let seq = [1, 2, 3, 4, 5];
        let out = truncate_sequence(&seq, 3, TruncationStrategy::Right).unwrap();
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn truncate_left() {
        let seq = [1, 2, 3, 4, 5];
        let out = truncate_sequence(&seq, 3, TruncationStrategy::Left).unwrap();
        assert_eq!(out, vec![3, 4, 5]);
    }

    #[test]
    fn truncate_center() {
        let seq = [1, 2, 3, 4, 5];
        let out = truncate_sequence(&seq, 3, TruncationStrategy::Center).unwrap();
        assert_eq!(out, vec![2, 3, 4]);
    }

    #[test]
    fn truncate_center_even_removal() {
        let seq = [1, 2, 3, 4, 5, 6];
        let out = truncate_sequence(&seq, 4, TruncationStrategy::Center).unwrap();
        assert_eq!(out, vec![2, 3, 4, 5]);
    }

    #[test]
    fn truncate_no_op_short() {
        let seq = [1, 2];
        let out = truncate_sequence(&seq, 5, TruncationStrategy::Right).unwrap();
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn truncate_exact_length() {
        let seq = [1, 2, 3];
        let out = truncate_sequence(&seq, 3, TruncationStrategy::Left).unwrap();
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn truncate_zero_max_len() {
        let err = truncate_sequence(&[1], 0, TruncationStrategy::Right).unwrap_err();
        assert!(err.to_string().contains("max_len must be > 0"));
    }

    #[test]
    fn truncate_to_one() {
        assert_eq!(
            truncate_sequence(&[10, 20, 30], 1, TruncationStrategy::Right).unwrap(),
            vec![10],
        );
        assert_eq!(
            truncate_sequence(&[10, 20, 30], 1, TruncationStrategy::Left).unwrap(),
            vec![30],
        );
        assert_eq!(
            truncate_sequence(&[10, 20, 30], 1, TruncationStrategy::Center).unwrap(),
            vec![20],
        );
    }

    // ── create_position_ids ────────────────────────────────────────

    #[test]
    fn position_ids_no_padding() {
        let ids = [10, 20, 30];
        let pos = create_position_ids(&ids, 0).unwrap();
        assert_eq!(pos, vec![0, 1, 2]);
    }

    #[test]
    fn position_ids_left_padded() {
        let ids = [0, 0, 5, 6]; // pad=0
        let pos = create_position_ids(&ids, 0).unwrap();
        assert_eq!(pos, vec![0, 0, 0, 1]);
    }

    #[test]
    fn position_ids_right_padded() {
        let ids = [5, 6, 0, 0]; // pad=0
        let pos = create_position_ids(&ids, 0).unwrap();
        assert_eq!(pos, vec![0, 1, 0, 0]);
    }

    #[test]
    fn position_ids_custom_pad() {
        let ids = [99, 1, 2, 99]; // pad=99
        let pos = create_position_ids(&ids, 99).unwrap();
        assert_eq!(pos, vec![0, 0, 1, 0]);
    }

    #[test]
    fn position_ids_empty() {
        let err = create_position_ids(&[], 0).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    // ── batch_encode_positions ─────────────────────────────────────

    #[test]
    fn batch_positions_basic() {
        // batch=2, seq=3, pad=0
        let ids = [0, 1, 2, 3, 0, 4];
        let pos = batch_encode_positions(&ids, 2, 3, 0).unwrap();
        assert_eq!(pos, vec![0, 0, 1, 0, 0, 1]);
    }

    #[test]
    fn batch_positions_no_padding() {
        let ids = [1, 2, 3, 4];
        let pos = batch_encode_positions(&ids, 2, 2, 0).unwrap();
        assert_eq!(pos, vec![0, 1, 0, 1]);
    }

    #[test]
    fn batch_positions_length_mismatch() {
        let err = batch_encode_positions(&[1, 2, 3], 2, 3, 0).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn batch_positions_zero_batch() {
        let err = batch_encode_positions(&[], 0, 3, 0).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    #[test]
    fn batch_positions_zero_seq() {
        let err = batch_encode_positions(&[], 2, 0, 0).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }

    // ── Integration / round-trip tests ─────────────────────────────

    #[test]
    fn pad_then_truncate_roundtrip() {
        let original = vec![1u32, 2, 3];
        let padded = right_pad_sequence(&original, 6, 0).unwrap();
        assert_eq!(padded.len(), 6);
        let truncated = truncate_sequence(&padded, 3, TruncationStrategy::Right).unwrap();
        assert_eq!(truncated, original);
    }

    #[test]
    fn left_pad_then_position_ids() {
        let seq = [5, 6, 7];
        let padded = left_pad_sequence(&seq, 5, 0).unwrap();
        let pos = create_position_ids(&padded, 0).unwrap();
        assert_eq!(pos, vec![0, 0, 0, 1, 2]);
    }

    #[test]
    fn causal_and_padding_mask_combined() {
        // seq_len=3, one padding at position 0
        let causal = create_causal_mask(3).unwrap();
        // Broadcast single-row padding mask to match causal shape.
        let pad_1d = [0.0, 1.0, 1.0]; // position 0 is padding
        let pad_2d: Vec<f32> = (0..3).flat_map(|_| pad_1d.iter().copied()).collect();
        let combined = combine_masks(&causal, &pad_2d).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 0.0, // row 0: causal allows [0], but pos 0 is pad
            0.0, 1.0, 0.0, // row 1: causal allows [0,1], but pos 0 is pad
            0.0, 1.0, 1.0, // row 2: causal allows [0,1,2], but pos 0 is pad
        ];
        assert_eq!(combined, expected);
    }

    #[test]
    fn one_hot_then_embedding_equivalence() {
        // one_hot @ embedding_table should equal direct lookup.
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens, dim 2
        let ids: &[u32] = &[2, 0];
        let direct = token_ids_to_embeddings(&table, 3, 2, ids).unwrap();
        let oh = one_hot_encode(ids, 3).unwrap();
        // Manual matmul: oh [2×3] @ table [3×2] = [2×2]
        let mut manual = vec![0.0f32; 2 * 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    manual[i * 2 + j] += oh[i * 3 + k] * table[k * 2 + j];
                }
            }
        }
        assert_eq!(direct, manual);
    }

    // ── Property tests ─────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn one_hot_correct_sum(
                n_tokens in 0..=20usize,
                vocab in 1..=64u32,
            ) {
                let ids: Vec<u32> = (0..n_tokens).map(|i| (i as u32) % vocab).collect();
                let oh = one_hot_encode(&ids, vocab).unwrap();
                let vs = vocab as usize;
                for row in 0..n_tokens {
                    let sum: f32 = oh[row * vs..(row + 1) * vs].iter().sum();
                    prop_assert!((sum - 1.0).abs() < f32::EPSILON);
                }
            }

            #[test]
            fn embedding_output_length(
                vocab in 1..=32u32,
                dim in 1..=16usize,
                n_ids in 0..=20usize,
            ) {
                let vs = vocab as usize;
                let table = vec![0.0f32; vs * dim];
                let ids: Vec<u32> = (0..n_ids).map(|i| (i as u32) % vocab).collect();
                let out = token_ids_to_embeddings(&table, vocab, dim, &ids).unwrap();
                prop_assert_eq!(out.len(), n_ids * dim);
            }

            #[test]
            fn causal_mask_row_sum(seq_len in 1..=32usize) {
                let mask = create_causal_mask(seq_len).unwrap();
                for i in 0..seq_len {
                    let sum: f32 = mask[i * seq_len..(i + 1) * seq_len].iter().sum();
                    prop_assert!((sum - (i + 1) as f32).abs() < f32::EPSILON);
                }
            }

            #[test]
            fn left_pad_preserves_content(
                len in 0..=20usize,
                target in 1..=30usize,
            ) {
                let seq: Vec<u32> = (1..=len as u32).collect();
                let padded = left_pad_sequence(&seq, target, 0).unwrap();
                // Original tokens should be a suffix of the result.
                if seq.len() < target {
                    prop_assert_eq!(&padded[target - seq.len()..], &seq[..]);
                } else {
                    prop_assert_eq!(padded, seq);
                }
            }

            #[test]
            fn right_pad_preserves_content(
                len in 0..=20usize,
                target in 1..=30usize,
            ) {
                let seq: Vec<u32> = (1..=len as u32).collect();
                let padded = right_pad_sequence(&seq, target, 0).unwrap();
                if seq.len() < target {
                    prop_assert_eq!(&padded[..seq.len()], &seq[..]);
                } else {
                    prop_assert_eq!(padded, seq);
                }
            }

            #[test]
            fn truncate_length_bound(
                len in 1..=30usize,
                max_len in 1..=30usize,
            ) {
                let seq: Vec<u32> = (0..len as u32).collect();
                for strat in [TruncationStrategy::Left, TruncationStrategy::Right, TruncationStrategy::Center] {
                    let out = truncate_sequence(&seq, max_len, strat).unwrap();
                    prop_assert!(out.len() <= max_len.max(seq.len()));
                    prop_assert!(out.len() <= seq.len());
                }
            }

            #[test]
            fn position_ids_max_equals_non_pad_count(
                len in 1..=20usize,
                pad_count in 0..=10usize,
            ) {
                let real = len.saturating_sub(pad_count);
                let mut seq: Vec<u32> = vec![0; pad_count.min(len)];
                seq.extend((1..=real as u32).take(len - seq.len()));
                seq.truncate(len);
                let pos = create_position_ids(&seq, 0).unwrap();
                let non_pad = seq.iter().filter(|&&t| t != 0).count();
                if non_pad > 0 {
                    let max_pos = *pos.iter().max().unwrap();
                    prop_assert_eq!(max_pos as usize, non_pad - 1);
                }
            }
        }
    }
}
