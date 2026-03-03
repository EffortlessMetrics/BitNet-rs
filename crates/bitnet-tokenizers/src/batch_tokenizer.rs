//! Batch tokenization utilities.
//!
//! Encode/decode multiple texts efficiently with padding support.

/// Padding strategy for batch tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad to the longest sequence in the batch.
    Longest,
    /// Pad to a fixed maximum length.
    MaxLength(usize),
    /// No padding.
    None,
}

/// Truncation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Truncate from the end.
    TruncateEnd(usize),
    /// Truncate from the start.
    TruncateStart(usize),
    /// No truncation.
    None,
}

/// A single tokenized sequence.
#[derive(Debug, Clone)]
pub struct TokenizedSequence {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub original_length: usize,
}

impl TokenizedSequence {
    pub fn new(ids: Vec<u32>) -> Self {
        let len = ids.len();
        Self { attention_mask: vec![1; len], input_ids: ids, original_length: len }
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
    pub fn padding_count(&self) -> usize {
        self.attention_mask.iter().filter(|&&m| m == 0).count()
    }
}

/// Batch of tokenized sequences.
#[derive(Debug, Clone)]
pub struct TokenBatch {
    pub sequences: Vec<TokenizedSequence>,
    pub max_length: usize,
}

impl TokenBatch {
    pub fn new(sequences: Vec<TokenizedSequence>) -> Self {
        let max_length = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        Self { sequences, max_length }
    }

    pub fn batch_size(&self) -> usize {
        self.sequences.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    pub fn total_tokens(&self) -> usize {
        self.sequences.iter().map(|s| s.original_length).sum()
    }

    pub fn total_padded_tokens(&self) -> usize {
        self.sequences.iter().map(|s| s.len()).sum()
    }

    pub fn padding_ratio(&self) -> f64 {
        let padded = self.total_padded_tokens();
        if padded == 0 {
            return 0.0;
        }
        let real = self.total_tokens();
        1.0 - (real as f64 / padded as f64)
    }
}

/// Apply padding to sequences.
pub fn pad_sequences(sequences: &mut [TokenizedSequence], strategy: PaddingStrategy, pad_id: u32) {
    let target = match strategy {
        PaddingStrategy::None => return,
        PaddingStrategy::Longest => sequences.iter().map(|s| s.len()).max().unwrap_or(0),
        PaddingStrategy::MaxLength(max) => max,
    };

    for seq in sequences.iter_mut() {
        while seq.input_ids.len() < target {
            seq.input_ids.push(pad_id);
            seq.attention_mask.push(0);
        }
    }
}

/// Apply truncation.
pub fn truncate_sequences(sequences: &mut [TokenizedSequence], strategy: TruncationStrategy) {
    match strategy {
        TruncationStrategy::None => {}
        TruncationStrategy::TruncateEnd(max) => {
            for seq in sequences.iter_mut() {
                if seq.input_ids.len() > max {
                    seq.input_ids.truncate(max);
                    seq.attention_mask.truncate(max);
                }
            }
        }
        TruncationStrategy::TruncateStart(max) => {
            for seq in sequences.iter_mut() {
                if seq.input_ids.len() > max {
                    let start = seq.input_ids.len() - max;
                    seq.input_ids = seq.input_ids[start..].to_vec();
                    seq.attention_mask = seq.attention_mask[start..].to_vec();
                }
            }
        }
    }
}

/// Build a batch from raw token id lists.
pub fn build_batch(
    token_lists: Vec<Vec<u32>>,
    padding: PaddingStrategy,
    truncation: TruncationStrategy,
    pad_id: u32,
) -> TokenBatch {
    let mut sequences: Vec<_> = token_lists.into_iter().map(TokenizedSequence::new).collect();
    truncate_sequences(&mut sequences, truncation);
    pad_sequences(&mut sequences, padding, pad_id);
    TokenBatch::new(sequences)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_new() {
        let seq = TokenizedSequence::new(vec![1, 2, 3]);
        assert_eq!(seq.len(), 3);
        assert_eq!(seq.padding_count(), 0);
        assert_eq!(seq.attention_mask, vec![1, 1, 1]);
    }

    #[test]
    fn test_pad_longest() {
        let mut seqs =
            vec![TokenizedSequence::new(vec![1, 2]), TokenizedSequence::new(vec![3, 4, 5])];
        pad_sequences(&mut seqs, PaddingStrategy::Longest, 0);
        assert_eq!(seqs[0].len(), 3);
        assert_eq!(seqs[0].padding_count(), 1);
    }

    #[test]
    fn test_pad_max_length() {
        let mut seqs = vec![TokenizedSequence::new(vec![1, 2])];
        pad_sequences(&mut seqs, PaddingStrategy::MaxLength(5), 0);
        assert_eq!(seqs[0].len(), 5);
    }

    #[test]
    fn test_pad_none() {
        let mut seqs = vec![TokenizedSequence::new(vec![1, 2])];
        pad_sequences(&mut seqs, PaddingStrategy::None, 0);
        assert_eq!(seqs[0].len(), 2);
    }

    #[test]
    fn test_truncate_end() {
        let mut seqs = vec![TokenizedSequence::new(vec![1, 2, 3, 4, 5])];
        truncate_sequences(&mut seqs, TruncationStrategy::TruncateEnd(3));
        assert_eq!(seqs[0].input_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_start() {
        let mut seqs = vec![TokenizedSequence::new(vec![1, 2, 3, 4, 5])];
        truncate_sequences(&mut seqs, TruncationStrategy::TruncateStart(3));
        assert_eq!(seqs[0].input_ids, vec![3, 4, 5]);
    }

    #[test]
    fn test_batch() {
        let batch = build_batch(
            vec![vec![1, 2], vec![3, 4, 5]],
            PaddingStrategy::Longest,
            TruncationStrategy::None,
            0,
        );
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.max_length, 3);
    }

    #[test]
    fn test_batch_padding_ratio() {
        let batch = build_batch(
            vec![vec![1], vec![1, 2, 3]],
            PaddingStrategy::Longest,
            TruncationStrategy::None,
            0,
        );
        assert!(batch.padding_ratio() > 0.0);
    }

    #[test]
    fn test_batch_total_tokens() {
        let batch = build_batch(
            vec![vec![1, 2], vec![3, 4, 5]],
            PaddingStrategy::None,
            TruncationStrategy::None,
            0,
        );
        assert_eq!(batch.total_tokens(), 5);
    }

    #[test]
    fn test_empty_batch() {
        let batch = build_batch(vec![], PaddingStrategy::None, TruncationStrategy::None, 0);
        assert!(batch.is_empty());
        assert_eq!(batch.padding_ratio(), 0.0);
    }

    #[test]
    fn test_truncate_and_pad() {
        let batch = build_batch(
            vec![vec![1, 2, 3, 4, 5], vec![6, 7]],
            PaddingStrategy::Longest,
            TruncationStrategy::TruncateEnd(3),
            0,
        );
        assert_eq!(batch.max_length, 3);
        assert_eq!(batch.sequences[1].padding_count(), 1);
    }

    #[test]
    fn test_sequence_empty() {
        let seq = TokenizedSequence::new(vec![]);
        assert!(seq.is_empty());
    }
}
