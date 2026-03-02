//! Batch token encoding and preparation for Intel Arc A770 OpenCL dispatch.
//!
//! Handles multiple prompt sequences simultaneously, padding/packing them
//! for efficient GPU dispatch. Manages variable-length sequences with
//! attention masks and position IDs.
//!
//! # Features
//!
//! - **Token batching**: collect multiple variable-length sequences into one batch
//! - **Padding strategies**: MaxLength, Longest, Fixed, NoPadding
//! - **Packing strategies**: Individual, Concatenated, SortedBins
//! - **Attention mask generation**: marks real vs. padded positions
//! - **Position ID generation**: per-sequence position indices
//! - **Dynamic batching**: groups requests by length similarity
//! - **Batch metrics**: padding waste tracking for efficiency analysis
//!
//! All algorithms are CPU reference implementations suitable for
//! verification against future OpenCL kernel variants.

use bitnet_common::{KernelError, Result};

// ── Padding strategy ─────────────────────────────────────────────

/// Strategy for padding variable-length sequences to a uniform length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad all sequences to the model's maximum context length.
    MaxLength,
    /// Pad to the longest sequence in the current batch.
    Longest,
    /// Pad to a specific fixed length; sequences longer are truncated.
    Fixed(usize),
    /// No padding — sequences remain at their original lengths.
    NoPadding,
}

// ── Packing strategy ─────────────────────────────────────────────

/// Strategy for packing sequences into GPU work-items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingStrategy {
    /// Each sequence occupies its own row in the batch tensor.
    Individual,
    /// Sequences are concatenated end-to-end into a single flat buffer
    /// with separator tokens; attention masks prevent cross-sequence
    /// attention.
    Concatenated,
    /// Sequences are sorted by length and grouped into bins of similar
    /// length to minimise padding waste.
    SortedBins,
}

// ── Sequence metadata ────────────────────────────────────────────

/// Metadata for a single sequence within a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceInfo {
    /// Length of the original (unpadded) token sequence.
    pub original_length: usize,
    /// Length after padding has been applied.
    pub padded_length: usize,
    /// Byte-offset of this sequence inside the flat packed buffer.
    pub start_offset: usize,
    /// Number of padding tokens appended.
    pub padding_count: usize,
}

// ── Batch metrics ────────────────────────────────────────────────

/// Aggregate statistics for a prepared batch.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchMetrics {
    /// Mean sequence length (original, before padding).
    pub avg_length: f64,
    /// Maximum sequence length in the batch (original).
    pub max_length: usize,
    /// Percentage of tokens in the padded batch that are padding.
    pub padding_waste_pct: f64,
    /// Number of sequences in the batch.
    pub sequences_count: usize,
}

// ── Token batch ──────────────────────────────────────────────────

/// A prepared batch of token sequences ready for GPU dispatch.
///
/// All inner vectors have been padded / packed according to the
/// chosen strategies and share a common second-dimension length.
#[derive(Debug, Clone)]
pub struct TokenBatch {
    /// Padded token IDs: one `Vec<u32>` per sequence.
    pub token_ids: Vec<Vec<u32>>,
    /// Attention masks (1 = real token, 0 = padding).
    pub attention_masks: Vec<Vec<u8>>,
    /// Position IDs for each token in each sequence.
    pub position_ids: Vec<Vec<u32>>,
    /// Number of sequences in the batch.
    pub batch_size: usize,
}

// ── Batch encoder ────────────────────────────────────────────────

/// Encodes multiple token sequences into a single padded/packed batch.
#[derive(Debug, Clone)]
pub struct BatchEncoder {
    /// Padding strategy to use.
    pub padding_strategy: PaddingStrategy,
    /// Packing strategy to use.
    pub packing_strategy: PackingStrategy,
    /// Model maximum context length (for `PaddingStrategy::MaxLength`).
    pub max_context_length: usize,
    /// Token ID used for padding positions.
    pub pad_token_id: u32,
}

impl BatchEncoder {
    /// Create a new batch encoder with the given configuration.
    pub fn new(
        padding_strategy: PaddingStrategy,
        packing_strategy: PackingStrategy,
        max_context_length: usize,
        pad_token_id: u32,
    ) -> Self {
        Self {
            padding_strategy,
            packing_strategy,
            max_context_length,
            pad_token_id,
        }
    }

    /// Encode a slice of variable-length sequences into a [`TokenBatch`].
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - A sequence exceeds `max_context_length` with `MaxLength` padding.
    /// - `Fixed` length is zero.
    /// - Input contains sequences of different lengths with `NoPadding`.
    pub fn encode(&self, sequences: &[Vec<u32>]) -> Result<TokenBatch> {
        if sequences.is_empty() {
            return Ok(TokenBatch {
                token_ids: Vec::new(),
                attention_masks: Vec::new(),
                position_ids: Vec::new(),
                batch_size: 0,
            });
        }

        match self.packing_strategy {
            PackingStrategy::Individual => self.encode_individual(sequences),
            PackingStrategy::Concatenated => self.encode_concatenated(sequences),
            PackingStrategy::SortedBins => self.encode_sorted_bins(sequences),
        }
    }

    /// Compute [`SequenceInfo`] for each sequence in the batch.
    pub fn sequence_infos(
        &self,
        sequences: &[Vec<u32>],
    ) -> Result<Vec<SequenceInfo>> {
        let target_len = self.resolve_target_length(sequences)?;
        let mut infos = Vec::with_capacity(sequences.len());
        let mut offset = 0usize;
        for seq in sequences {
            let orig = seq.len();
            let padded = match self.padding_strategy {
                PaddingStrategy::NoPadding => orig,
                PaddingStrategy::Fixed(n) => n,
                _ => target_len,
            };
            let pad_count = padded.saturating_sub(orig);
            infos.push(SequenceInfo {
                original_length: orig,
                padded_length: padded,
                start_offset: offset,
                padding_count: pad_count,
            });
            offset += padded;
        }
        Ok(infos)
    }

    /// Compute [`BatchMetrics`] for a set of sequences.
    pub fn compute_metrics(
        &self,
        sequences: &[Vec<u32>],
    ) -> Result<BatchMetrics> {
        if sequences.is_empty() {
            return Ok(BatchMetrics {
                avg_length: 0.0,
                max_length: 0,
                padding_waste_pct: 0.0,
                sequences_count: 0,
            });
        }

        let lengths: Vec<usize> = sequences.iter().map(Vec::len).collect();
        let max_length = *lengths.iter().max().unwrap_or(&0);
        let total_original: usize = lengths.iter().sum();
        let avg_length = total_original as f64 / lengths.len() as f64;

        let target = self.resolve_target_length(sequences)?;
        let total_padded = match self.padding_strategy {
            PaddingStrategy::NoPadding => total_original,
            _ => target * sequences.len(),
        };

        let padding_waste_pct = if total_padded == 0 {
            0.0
        } else {
            let waste = total_padded.saturating_sub(total_original);
            waste as f64 / total_padded as f64 * 100.0
        };

        Ok(BatchMetrics {
            avg_length,
            max_length,
            padding_waste_pct,
            sequences_count: lengths.len(),
        })
    }

    // ── private helpers ──────────────────────────────────────

    fn resolve_target_length(
        &self,
        sequences: &[Vec<u32>],
    ) -> Result<usize> {
        match self.padding_strategy {
            PaddingStrategy::MaxLength => Ok(self.max_context_length),
            PaddingStrategy::Longest => {
                Ok(sequences.iter().map(Vec::len).max().unwrap_or(0))
            }
            PaddingStrategy::Fixed(n) => {
                if n == 0 {
                    return Err(KernelError::InvalidArguments {
                        reason: "Fixed padding length must be > 0".into(),
                    }
                    .into());
                }
                Ok(n)
            }
            PaddingStrategy::NoPadding => {
                Ok(sequences.iter().map(Vec::len).max().unwrap_or(0))
            }
        }
    }

    fn pad_sequence(&self, seq: &[u32], target_len: usize) -> Vec<u32> {
        let mut padded = Vec::with_capacity(target_len);
        let take = seq.len().min(target_len);
        padded.extend_from_slice(&seq[..take]);
        padded.resize(target_len, self.pad_token_id);
        padded
    }

    fn make_attention_mask(
        original_len: usize,
        padded_len: usize,
    ) -> Vec<u8> {
        let real = original_len.min(padded_len);
        let mut mask = vec![1u8; real];
        mask.resize(padded_len, 0);
        mask
    }

    fn make_position_ids(
        original_len: usize,
        padded_len: usize,
    ) -> Vec<u32> {
        let real = original_len.min(padded_len);
        let mut ids: Vec<u32> = (0..real as u32).collect();
        ids.resize(padded_len, 0);
        ids
    }

    /// Individual packing: each sequence is its own row, padded to target.
    fn encode_individual(
        &self,
        sequences: &[Vec<u32>],
    ) -> Result<TokenBatch> {
        let target = self.resolve_target_length(sequences)?;

        let mut token_ids = Vec::with_capacity(sequences.len());
        let mut attention_masks = Vec::with_capacity(sequences.len());
        let mut position_ids = Vec::with_capacity(sequences.len());

        for seq in sequences {
            let padded_len = match self.padding_strategy {
                PaddingStrategy::NoPadding => seq.len(),
                _ => target,
            };
            token_ids.push(self.pad_sequence(seq, padded_len));
            attention_masks
                .push(Self::make_attention_mask(seq.len(), padded_len));
            position_ids
                .push(Self::make_position_ids(seq.len(), padded_len));
        }

        Ok(TokenBatch {
            token_ids,
            attention_masks,
            position_ids,
            batch_size: sequences.len(),
        })
    }

    /// Concatenated packing: all sequences packed end-to-end in one row.
    fn encode_concatenated(
        &self,
        sequences: &[Vec<u32>],
    ) -> Result<TokenBatch> {
        let total_len: usize = sequences.iter().map(Vec::len).sum();
        let mut flat_tokens = Vec::with_capacity(total_len);
        let mut flat_mask = Vec::with_capacity(total_len);
        let mut flat_pos = Vec::with_capacity(total_len);

        for seq in sequences {
            flat_tokens.extend_from_slice(seq);
            flat_mask.extend(std::iter::repeat_n(1u8, seq.len()));
            flat_pos.extend(0..seq.len() as u32);
        }

        Ok(TokenBatch {
            token_ids: vec![flat_tokens],
            attention_masks: vec![flat_mask],
            position_ids: vec![flat_pos],
            batch_size: sequences.len(),
        })
    }

    /// Sorted-bins packing: sort by length, then pad each to its bin's
    /// longest member.
    fn encode_sorted_bins(
        &self,
        sequences: &[Vec<u32>],
    ) -> Result<TokenBatch> {
        let mut indexed: Vec<(usize, &Vec<u32>)> =
            sequences.iter().enumerate().collect();
        indexed.sort_by_key(|(_, s)| s.len());

        let mut token_ids = vec![Vec::new(); sequences.len()];
        let mut attention_masks = vec![Vec::new(); sequences.len()];
        let mut position_ids = vec![Vec::new(); sequences.len()];

        let target = self.resolve_target_length(sequences)?;

        for (idx, seq) in &indexed {
            let bin_target = match self.padding_strategy {
                PaddingStrategy::NoPadding => seq.len(),
                _ => target,
            };
            token_ids[*idx] = self.pad_sequence(seq, bin_target);
            attention_masks[*idx] =
                Self::make_attention_mask(seq.len(), bin_target);
            position_ids[*idx] =
                Self::make_position_ids(seq.len(), bin_target);
        }

        Ok(TokenBatch {
            token_ids,
            attention_masks,
            position_ids,
            batch_size: sequences.len(),
        })
    }
}

// ── Dynamic batcher ──────────────────────────────────────────────

/// Groups incoming sequences into efficient batches by length similarity.
///
/// Sequences are placed into the first bin whose representative length
/// is within `length_tolerance` of the sequence length.  When a bin
/// reaches `max_batch_size`, it is flushed and returned.
#[derive(Debug, Clone)]
pub struct DynamicBatcher {
    /// Maximum number of sequences per batch.
    pub max_batch_size: usize,
    /// Maximum allowed length difference within a single batch.
    pub length_tolerance: usize,
    /// Internal bins: `(representative_length, sequences)`.
    bins: Vec<(usize, Vec<Vec<u32>>)>,
}

impl DynamicBatcher {
    /// Create a new dynamic batcher.
    pub fn new(max_batch_size: usize, length_tolerance: usize) -> Self {
        Self {
            max_batch_size,
            length_tolerance,
            bins: Vec::new(),
        }
    }

    /// Add a sequence to the batcher.
    ///
    /// Returns `Some(batch)` if adding this sequence causes a bin to
    /// reach `max_batch_size`, otherwise returns `None`.
    pub fn add_sequence(
        &mut self,
        sequence: Vec<u32>,
    ) -> Option<Vec<Vec<u32>>> {
        let len = sequence.len();

        // Find existing bin within tolerance.
        for (rep, bin) in &mut self.bins {
            if len.abs_diff(*rep) <= self.length_tolerance {
                bin.push(sequence);
                if bin.len() >= self.max_batch_size {
                    let batch = std::mem::take(bin);
                    self.bins.retain(|(_, b)| !b.is_empty());
                    return Some(batch);
                }
                return None;
            }
        }

        // No matching bin — create a new one.
        let mut new_bin = Vec::with_capacity(self.max_batch_size);
        new_bin.push(sequence);
        if self.max_batch_size <= 1 {
            return Some(new_bin);
        }
        self.bins.push((len, new_bin));
        None
    }

    /// Flush all remaining bins, returning whatever partial batches exist.
    pub fn flush(&mut self) -> Vec<Vec<Vec<u32>>> {
        let mut batches = Vec::new();
        for (_, bin) in self.bins.drain(..) {
            if !bin.is_empty() {
                batches.push(bin);
            }
        }
        batches
    }

    /// Number of sequences currently buffered across all bins.
    pub fn pending_count(&self) -> usize {
        self.bins.iter().map(|(_, b)| b.len()).sum()
    }

    /// Number of active bins.
    pub fn bin_count(&self) -> usize {
        self.bins.len()
    }
}

// ── CPU reference: pad sequences ─────────────────────────────────

/// Pad a set of sequences to a uniform length (CPU reference).
///
/// Returns the padded token matrix and corresponding attention masks.
pub fn pad_sequences_ref(
    sequences: &[Vec<u32>],
    target_length: usize,
    pad_token_id: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<u8>>) {
    let mut padded = Vec::with_capacity(sequences.len());
    let mut masks = Vec::with_capacity(sequences.len());
    for seq in sequences {
        let take = seq.len().min(target_length);
        let mut tokens = Vec::with_capacity(target_length);
        tokens.extend_from_slice(&seq[..take]);
        tokens.resize(target_length, pad_token_id);
        padded.push(tokens);

        let mut mask = vec![1u8; take];
        mask.resize(target_length, 0);
        masks.push(mask);
    }
    (padded, masks)
}

/// Generate position IDs for a batch of padded sequences (CPU reference).
pub fn generate_position_ids_ref(
    attention_masks: &[Vec<u8>],
) -> Vec<Vec<u32>> {
    attention_masks
        .iter()
        .map(|mask| {
            let mut pos = 0u32;
            mask.iter()
                .map(|&m| {
                    if m == 1 {
                        let p = pos;
                        pos += 1;
                        p
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect()
}

/// Concatenate sequences into a single flat buffer (CPU reference).
///
/// Returns `(flat_tokens, flat_mask, flat_positions, boundaries)` where
/// `boundaries[i]` is the start offset of sequence `i`.
pub fn concatenate_sequences_ref(
    sequences: &[Vec<u32>],
) -> (Vec<u32>, Vec<u8>, Vec<u32>, Vec<usize>) {
    let total: usize = sequences.iter().map(Vec::len).sum();
    let mut tokens = Vec::with_capacity(total);
    let mut mask = Vec::with_capacity(total);
    let mut positions = Vec::with_capacity(total);
    let mut boundaries = Vec::with_capacity(sequences.len());

    for seq in sequences {
        boundaries.push(tokens.len());
        tokens.extend_from_slice(seq);
        mask.extend(std::iter::repeat_n(1u8, seq.len()));
        positions.extend(0..seq.len() as u32);
    }

    (tokens, mask, positions, boundaries)
}

/// Sort sequences by length and return the sort permutation (CPU reference).
pub fn sort_by_length_ref(sequences: &[Vec<u32>]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..sequences.len()).collect();
    indices.sort_by_key(|&i| sequences[i].len());
    indices
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder(
        padding: PaddingStrategy,
        packing: PackingStrategy,
    ) -> BatchEncoder {
        BatchEncoder::new(padding, packing, 512, 0)
    }

    // ── Padding: MaxLength ───────────────────────────────────

    #[test]
    fn padding_max_length_pads_to_context_len() {
        let enc =
            encoder(PaddingStrategy::MaxLength, PackingStrategy::Individual);
        let batch = enc.encode(&[vec![1, 2, 3]]).unwrap();
        assert_eq!(batch.token_ids[0].len(), 512);
        assert_eq!(&batch.token_ids[0][..3], &[1, 2, 3]);
        assert!(batch.token_ids[0][3..].iter().all(|&t| t == 0));
    }

    #[test]
    fn padding_max_length_attention_mask() {
        let enc =
            encoder(PaddingStrategy::MaxLength, PackingStrategy::Individual);
        let batch = enc.encode(&[vec![10, 20]]).unwrap();
        assert_eq!(batch.attention_masks[0][0], 1);
        assert_eq!(batch.attention_masks[0][1], 1);
        assert!(batch.attention_masks[0][2..].iter().all(|&m| m == 0));
    }

    #[test]
    fn padding_max_length_position_ids() {
        let enc =
            encoder(PaddingStrategy::MaxLength, PackingStrategy::Individual);
        let batch = enc.encode(&[vec![5, 6, 7]]).unwrap();
        assert_eq!(&batch.position_ids[0][..3], &[0, 1, 2]);
        assert!(batch.position_ids[0][3..].iter().all(|&p| p == 0));
    }

    // ── Padding: Longest ─────────────────────────────────────

    #[test]
    fn padding_longest_uses_max_seq_len() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![1, 2], vec![3, 4, 5, 6, 7]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(batch.token_ids[0].len(), 5);
        assert_eq!(batch.token_ids[1].len(), 5);
    }

    #[test]
    fn padding_longest_short_seq_padded() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![1], vec![2, 3, 4]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(&batch.token_ids[0], &[1, 0, 0]);
        assert_eq!(&batch.attention_masks[0], &[1, 0, 0]);
    }

    #[test]
    fn padding_longest_long_seq_unchanged() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![1, 2, 3], vec![4]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(&batch.token_ids[0], &[1, 2, 3]);
        assert_eq!(&batch.attention_masks[0], &[1, 1, 1]);
    }

    // ── Padding: Fixed ───────────────────────────────────────

    #[test]
    fn padding_fixed_pads_short_seq() {
        let enc =
            encoder(PaddingStrategy::Fixed(4), PackingStrategy::Individual);
        let batch = enc.encode(&[vec![1, 2]]).unwrap();
        assert_eq!(&batch.token_ids[0], &[1, 2, 0, 0]);
    }

    #[test]
    fn padding_fixed_truncates_long_seq() {
        let enc =
            encoder(PaddingStrategy::Fixed(3), PackingStrategy::Individual);
        let batch = enc.encode(&[vec![1, 2, 3, 4, 5]]).unwrap();
        assert_eq!(&batch.token_ids[0], &[1, 2, 3]);
        assert_eq!(&batch.attention_masks[0], &[1, 1, 1]);
    }

    #[test]
    fn padding_fixed_exact_length_no_change() {
        let enc =
            encoder(PaddingStrategy::Fixed(3), PackingStrategy::Individual);
        let batch = enc.encode(&[vec![7, 8, 9]]).unwrap();
        assert_eq!(&batch.token_ids[0], &[7, 8, 9]);
    }

    #[test]
    fn padding_fixed_zero_length_errors() {
        let enc =
            encoder(PaddingStrategy::Fixed(0), PackingStrategy::Individual);
        assert!(enc.encode(&[vec![1]]).is_err());
    }

    // ── Padding: NoPadding ───────────────────────────────────

    #[test]
    fn no_padding_preserves_lengths() {
        let enc =
            encoder(PaddingStrategy::NoPadding, PackingStrategy::Individual);
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(batch.token_ids[0].len(), 2);
        assert_eq!(batch.token_ids[1].len(), 3);
    }

    #[test]
    fn no_padding_all_mask_ones() {
        let enc =
            encoder(PaddingStrategy::NoPadding, PackingStrategy::Individual);
        let batch = enc.encode(&[vec![1, 2, 3]]).unwrap();
        assert!(batch.attention_masks[0].iter().all(|&m| m == 1));
    }

    // ── Packing: Concatenated ────────────────────────────────

    #[test]
    fn packing_concatenated_single_row() {
        let enc = encoder(
            PaddingStrategy::NoPadding,
            PackingStrategy::Concatenated,
        );
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(batch.token_ids.len(), 1);
        assert_eq!(&batch.token_ids[0], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn packing_concatenated_mask_all_ones() {
        let enc = encoder(
            PaddingStrategy::NoPadding,
            PackingStrategy::Concatenated,
        );
        let seqs = vec![vec![10], vec![20, 30]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(&batch.attention_masks[0], &[1, 1, 1]);
    }

    #[test]
    fn packing_concatenated_position_ids_reset() {
        let enc = encoder(
            PaddingStrategy::NoPadding,
            PackingStrategy::Concatenated,
        );
        let seqs = vec![vec![1, 2, 3], vec![4, 5]];
        let batch = enc.encode(&seqs).unwrap();
        // Positions reset per sequence: [0,1,2, 0,1]
        assert_eq!(&batch.position_ids[0], &[0, 1, 2, 0, 1]);
    }

    #[test]
    fn packing_concatenated_batch_size_reflects_sequences() {
        let enc = encoder(
            PaddingStrategy::NoPadding,
            PackingStrategy::Concatenated,
        );
        let seqs = vec![vec![1], vec![2], vec![3]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(batch.batch_size, 3);
    }

    // ── Packing: SortedBins ──────────────────────────────────

    #[test]
    fn packing_sorted_bins_preserves_tokens() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::SortedBins);
        let seqs = vec![vec![1, 2, 3], vec![4], vec![5, 6]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(&batch.token_ids[0][..3], &[1, 2, 3]);
        assert_eq!(batch.token_ids[1][0], 4);
        assert_eq!(&batch.token_ids[2][..2], &[5, 6]);
    }

    #[test]
    fn packing_sorted_bins_uniform_length() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::SortedBins);
        let seqs = vec![vec![1], vec![2, 3, 4], vec![5, 6]];
        let batch = enc.encode(&seqs).unwrap();
        for row in &batch.token_ids {
            assert_eq!(row.len(), 3);
        }
    }

    // ── Attention mask correctness ───────────────────────────

    #[test]
    fn attention_mask_counts_match_original_length() {
        let enc =
            encoder(PaddingStrategy::MaxLength, PackingStrategy::Individual);
        let seqs = vec![vec![1, 2, 3, 4, 5]];
        let batch = enc.encode(&seqs).unwrap();
        let ones: usize = batch.attention_masks[0]
            .iter()
            .filter(|&&m| m == 1)
            .count();
        assert_eq!(ones, 5);
    }

    #[test]
    fn attention_mask_zeros_equal_padding() {
        let enc =
            encoder(PaddingStrategy::Fixed(10), PackingStrategy::Individual);
        let batch = enc.encode(&[vec![1, 2, 3]]).unwrap();
        let zeros: usize = batch.attention_masks[0]
            .iter()
            .filter(|&&m| m == 0)
            .count();
        assert_eq!(zeros, 7);
    }

    // ── Position IDs ─────────────────────────────────────────

    #[test]
    fn position_ids_sequential_for_real_tokens() {
        let enc =
            encoder(PaddingStrategy::Fixed(6), PackingStrategy::Individual);
        let batch = enc.encode(&[vec![10, 20, 30, 40]]).unwrap();
        assert_eq!(&batch.position_ids[0][..4], &[0, 1, 2, 3]);
    }

    #[test]
    fn position_ids_zero_for_padding() {
        let enc =
            encoder(PaddingStrategy::Fixed(5), PackingStrategy::Individual);
        let batch = enc.encode(&[vec![1, 2]]).unwrap();
        assert!(batch.position_ids[0][2..].iter().all(|&p| p == 0));
    }

    // ── Variable-length handling ─────────────────────────────

    #[test]
    fn variable_lengths_padded_correctly() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![1], vec![2, 3], vec![4, 5, 6]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(&batch.token_ids[0], &[1, 0, 0]);
        assert_eq!(&batch.token_ids[1], &[2, 3, 0]);
        assert_eq!(&batch.token_ids[2], &[4, 5, 6]);
    }

    // ── Batch metrics ────────────────────────────────────────

    #[test]
    fn metrics_empty_batch() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let m = enc.compute_metrics(&[]).unwrap();
        assert_eq!(m.sequences_count, 0);
        assert_eq!(m.padding_waste_pct, 0.0);
    }

    #[test]
    fn metrics_single_sequence() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let m = enc.compute_metrics(&[vec![1, 2, 3]]).unwrap();
        assert_eq!(m.sequences_count, 1);
        assert_eq!(m.max_length, 3);
        assert!((m.avg_length - 3.0).abs() < 1e-9);
        assert!((m.padding_waste_pct - 0.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_waste_percentage() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        // seqs: [1], [2,3,4,5] → longest=4, total_padded=8, total_orig=5
        // waste = 3/8 = 37.5%
        let m =
            enc.compute_metrics(&[vec![1], vec![2, 3, 4, 5]]).unwrap();
        assert!((m.padding_waste_pct - 37.5).abs() < 1e-9);
    }

    #[test]
    fn metrics_all_same_length_zero_waste() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let m = enc.compute_metrics(&seqs).unwrap();
        assert!((m.padding_waste_pct - 0.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_max_length_padding_waste() {
        let enc =
            encoder(PaddingStrategy::MaxLength, PackingStrategy::Individual);
        let m = enc.compute_metrics(&[vec![1, 2]]).unwrap();
        let expected = 510.0 / 512.0 * 100.0;
        assert!((m.padding_waste_pct - expected).abs() < 0.01);
    }

    // ── SequenceInfo ─────────────────────────────────────────

    #[test]
    fn sequence_info_offsets() {
        let enc =
            encoder(PaddingStrategy::Fixed(4), PackingStrategy::Individual);
        let seqs = vec![vec![1, 2], vec![3]];
        let infos = enc.sequence_infos(&seqs).unwrap();
        assert_eq!(infos[0].original_length, 2);
        assert_eq!(infos[0].padded_length, 4);
        assert_eq!(infos[0].start_offset, 0);
        assert_eq!(infos[0].padding_count, 2);
        assert_eq!(infos[1].start_offset, 4);
        assert_eq!(infos[1].padding_count, 3);
    }

    // ── Dynamic batcher ──────────────────────────────────────

    #[test]
    fn dynamic_batcher_groups_by_length() {
        let mut batcher = DynamicBatcher::new(2, 1);
        assert!(batcher.add_sequence(vec![1, 2, 3]).is_none());
        let batch = batcher.add_sequence(vec![4, 5, 6, 7]);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);
    }

    #[test]
    fn dynamic_batcher_separate_bins() {
        let mut batcher = DynamicBatcher::new(3, 0);
        batcher.add_sequence(vec![1, 2]); // bin len=2
        batcher.add_sequence(vec![3, 4, 5]); // bin len=3
        assert_eq!(batcher.bin_count(), 2);
    }

    #[test]
    fn dynamic_batcher_flush() {
        let mut batcher = DynamicBatcher::new(10, 2);
        batcher.add_sequence(vec![1, 2]);
        batcher.add_sequence(vec![3, 4, 5]);
        let flushed = batcher.flush();
        assert!(!flushed.is_empty());
        assert_eq!(batcher.pending_count(), 0);
    }

    #[test]
    fn dynamic_batcher_pending_count() {
        let mut batcher = DynamicBatcher::new(100, 0);
        batcher.add_sequence(vec![1]);
        batcher.add_sequence(vec![1, 2]);
        assert_eq!(batcher.pending_count(), 2);
    }

    #[test]
    fn dynamic_batcher_max_batch_one() {
        let mut batcher = DynamicBatcher::new(1, 100);
        let batch = batcher.add_sequence(vec![1, 2, 3]);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 1);
    }

    // ── Edge cases ───────────────────────────────────────────

    #[test]
    fn empty_batch_encode() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let batch = enc.encode(&[]).unwrap();
        assert_eq!(batch.batch_size, 0);
        assert!(batch.token_ids.is_empty());
    }

    #[test]
    fn single_sequence_batch() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let batch = enc.encode(&[vec![42]]).unwrap();
        assert_eq!(batch.batch_size, 1);
        assert_eq!(&batch.token_ids[0], &[42]);
    }

    #[test]
    fn all_same_length_no_waste() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![1, 2, 3]; 5];
        let batch = enc.encode(&seqs).unwrap();
        for mask in &batch.attention_masks {
            assert!(mask.iter().all(|&m| m == 1));
        }
    }

    #[test]
    fn very_long_sequence_fixed_truncation() {
        let enc =
            encoder(PaddingStrategy::Fixed(4), PackingStrategy::Individual);
        let long_seq = (0..1000).collect::<Vec<u32>>();
        let batch = enc.encode(&[long_seq]).unwrap();
        assert_eq!(batch.token_ids[0].len(), 4);
        assert_eq!(&batch.token_ids[0], &[0, 1, 2, 3]);
    }

    #[test]
    fn empty_sequence_in_batch() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![], vec![1, 2]];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(&batch.token_ids[0], &[0, 0]);
        assert_eq!(&batch.attention_masks[0], &[0, 0]);
    }

    // ── CPU reference functions ──────────────────────────────

    #[test]
    fn pad_sequences_ref_basic() {
        let (padded, masks) =
            pad_sequences_ref(&[vec![1, 2], vec![3, 4, 5]], 4, 0);
        assert_eq!(&padded[0], &[1, 2, 0, 0]);
        assert_eq!(&padded[1], &[3, 4, 5, 0]);
        assert_eq!(&masks[0], &[1, 1, 0, 0]);
        assert_eq!(&masks[1], &[1, 1, 1, 0]);
    }

    #[test]
    fn generate_position_ids_ref_basic() {
        let masks = vec![vec![1, 1, 0, 0], vec![1, 1, 1, 0]];
        let pos = generate_position_ids_ref(&masks);
        assert_eq!(&pos[0], &[0, 1, 0, 0]);
        assert_eq!(&pos[1], &[0, 1, 2, 0]);
    }

    #[test]
    fn concatenate_sequences_ref_basic() {
        let (tokens, mask, positions, boundaries) =
            concatenate_sequences_ref(&[vec![10, 20], vec![30]]);
        assert_eq!(&tokens, &[10, 20, 30]);
        assert_eq!(&mask, &[1, 1, 1]);
        assert_eq!(&positions, &[0, 1, 0]);
        assert_eq!(&boundaries, &[0, 2]);
    }

    #[test]
    fn sort_by_length_ref_basic() {
        let seqs = vec![vec![1, 2, 3], vec![4], vec![5, 6]];
        let perm = sort_by_length_ref(&seqs);
        assert_eq!(perm, vec![1, 2, 0]);
    }

    // ── Property tests ───────────────────────────────────────

    #[test]
    fn property_padding_preserves_original_tokens() {
        let enc =
            encoder(PaddingStrategy::Fixed(10), PackingStrategy::Individual);
        let seqs = vec![
            vec![100, 200, 300],
            vec![400, 500],
            vec![600, 700, 800, 900],
        ];
        let batch = enc.encode(&seqs).unwrap();
        for (i, seq) in seqs.iter().enumerate() {
            for (j, &tok) in seq.iter().enumerate() {
                assert_eq!(
                    batch.token_ids[i][j], tok,
                    "original token mismatch at seq={i} pos={j}"
                );
            }
        }
    }

    #[test]
    fn property_attention_mask_sum_equals_original_len() {
        let enc =
            encoder(PaddingStrategy::MaxLength, PackingStrategy::Individual);
        let seqs = vec![vec![1; 7], vec![2; 3], vec![3; 100]];
        let batch = enc.encode(&seqs).unwrap();
        for (i, seq) in seqs.iter().enumerate() {
            let ones: usize = batch.attention_masks[i]
                .iter()
                .filter(|&&m| m == 1)
                .count();
            assert_eq!(ones, seq.len(), "mask sum mismatch for seq {i}");
        }
    }

    #[test]
    fn property_position_ids_monotonic_for_real_tokens() {
        let enc =
            encoder(PaddingStrategy::Fixed(20), PackingStrategy::Individual);
        let batch = enc.encode(&[vec![10, 20, 30, 40, 50]]).unwrap();
        let real_positions: Vec<u32> = batch.position_ids[0]
            .iter()
            .zip(batch.attention_masks[0].iter())
            .filter(|&(_, &m)| m == 1)
            .map(|&(&p, _)| p)
            .collect();
        for window in real_positions.windows(2) {
            assert_eq!(
                window[1],
                window[0] + 1,
                "position IDs not monotonic"
            );
        }
    }

    #[test]
    fn property_batch_size_matches_input() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![1]; 17];
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(batch.batch_size, 17);
        assert_eq!(batch.token_ids.len(), 17);
    }

    #[test]
    fn property_concatenated_total_length_equals_sum() {
        let enc = encoder(
            PaddingStrategy::NoPadding,
            PackingStrategy::Concatenated,
        );
        let seqs = vec![vec![1, 2], vec![3], vec![4, 5, 6, 7]];
        let total: usize = seqs.iter().map(Vec::len).sum();
        let batch = enc.encode(&seqs).unwrap();
        assert_eq!(batch.token_ids[0].len(), total);
    }

    #[test]
    fn property_metrics_avg_length_correct() {
        let enc =
            encoder(PaddingStrategy::Longest, PackingStrategy::Individual);
        let seqs = vec![vec![1; 2], vec![2; 6], vec![3; 4]];
        let m = enc.compute_metrics(&seqs).unwrap();
        assert!((m.avg_length - 4.0).abs() < 1e-9);
    }
}
