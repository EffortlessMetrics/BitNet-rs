//! Batch processor for efficient inference preparation.
//!
//! Groups and pads tokenized inputs into `InferenceBatch` structs
//! ready for the inference engine, with optional dynamic batching
//! that reorders requests by sequence length for efficiency.

use crate::tokenizer_gpu::GpuTokenizer;

/// A prepared batch of inputs ready for inference.
#[derive(Debug, Clone)]
pub struct InferenceBatch {
    /// Padded token-id matrix (`batch_size` × `max_seq_len`).
    pub input_ids: Vec<Vec<u32>>,
    /// Binary attention mask (`batch_size` × `max_seq_len`).
    pub attention_mask: Vec<Vec<u8>>,
    /// Position ids (`batch_size` × `max_seq_len`).
    pub positions: Vec<Vec<u32>>,
    /// Number of sequences in this batch.
    pub batch_size: usize,
}

impl InferenceBatch {
    /// Maximum sequence length in this batch.
    pub fn max_seq_len(&self) -> usize {
        self.input_ids.first().map_or(0, Vec::len)
    }

    /// True when the batch contains no sequences.
    pub const fn is_empty(&self) -> bool {
        self.batch_size == 0
    }
}

/// Batch processor that converts raw text into inference-ready batches.
pub struct BatchProcessor {
    pad_id: u32,
}

impl BatchProcessor {
    /// Create a new `BatchProcessor` with the given pad token id.
    pub const fn new(pad_id: u32) -> Self {
        Self { pad_id }
    }

    /// Tokenize, pad, and build an `InferenceBatch` from raw texts.
    pub fn prepare_batch(&self, texts: &[&str], tokenizer: &GpuTokenizer) -> InferenceBatch {
        if texts.is_empty() {
            return InferenceBatch {
                input_ids: Vec::new(),
                attention_mask: Vec::new(),
                positions: Vec::new(),
                batch_size: 0,
            };
        }

        let encoded = tokenizer.encode_batch(texts);
        let (padded, lengths) = tokenizer.pad_sequences(&encoded, self.pad_id);

        let max_len = padded.first().map_or(0, Vec::len);
        let mask = GpuTokenizer::create_attention_mask(&lengths, max_len);

        #[allow(clippy::cast_possible_truncation)] // position indices fit in u32
        let positions: Vec<Vec<u32>> = lengths
            .iter()
            .map(|&len| (0..max_len).map(|i| if i < len { i as u32 } else { 0 }).collect())
            .collect();

        InferenceBatch {
            batch_size: padded.len(),
            input_ids: padded,
            attention_mask: mask,
            positions,
        }
    }

    /// Split a large batch into smaller sub-batches of at most
    /// `max_size` sequences each.
    pub fn split_batch(batch: InferenceBatch, max_size: usize) -> Vec<InferenceBatch> {
        if max_size == 0 || batch.batch_size <= max_size {
            return vec![batch];
        }

        let mut result = Vec::new();
        let mut start = 0;
        while start < batch.batch_size {
            let end = (start + max_size).min(batch.batch_size);
            result.push(InferenceBatch {
                input_ids: batch.input_ids[start..end].to_vec(),
                attention_mask: batch.attention_mask[start..end].to_vec(),
                positions: batch.positions[start..end].to_vec(),
                batch_size: end - start,
            });
            start = end;
        }
        result
    }

    /// Reorder texts by token length so that similarly-sized sequences
    /// end up in the same sub-batch, reducing padding waste.
    ///
    /// Returns `(sorted_texts, original_indices)` so callers can
    /// unsort the results later.
    pub fn dynamic_batch_sort<'a>(texts: &[&'a str]) -> (Vec<&'a str>, Vec<usize>) {
        let mut indexed: Vec<(usize, &&str)> = texts.iter().enumerate().collect();
        indexed.sort_by_key(|(_, t)| t.len());
        let sorted: Vec<&str> = indexed.iter().map(|(_, t)| **t).collect();
        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        (sorted, indices)
    }

    /// Unsort a vec of results using the index map returned by
    /// `dynamic_batch_sort`.
    pub fn unsort<T: Clone>(sorted: &[T], indices: &[usize]) -> Vec<T> {
        let mut out = sorted.to_vec();
        for (sorted_pos, &orig_idx) in indices.iter().enumerate() {
            if orig_idx < out.len() {
                out[orig_idx] = sorted[sorted_pos].clone();
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer_gpu::{GpuTokenizer, GpuTokenizerConfig};

    fn make_tokenizer() -> GpuTokenizer {
        GpuTokenizer::new(GpuTokenizerConfig { max_length: 128, ..Default::default() })
    }

    #[test]
    fn prepare_empty_batch() {
        let proc = BatchProcessor::new(0);
        let tok = make_tokenizer();
        let batch = proc.prepare_batch(&[], &tok);
        assert!(batch.is_empty());
        assert_eq!(batch.batch_size, 0);
    }

    #[test]
    fn prepare_single_item() {
        let proc = BatchProcessor::new(0);
        let tok = make_tokenizer();
        let batch = proc.prepare_batch(&["hi"], &tok);
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.input_ids[0], vec![104, 105]);
    }

    #[test]
    fn split_batch_noop_when_small() {
        let proc = BatchProcessor::new(0);
        let tok = make_tokenizer();
        let batch = proc.prepare_batch(&["a", "b"], &tok);
        let splits = BatchProcessor::split_batch(batch, 10);
        assert_eq!(splits.len(), 1);
    }

    #[test]
    fn split_batch_divides_evenly() {
        let proc = BatchProcessor::new(0);
        let tok = make_tokenizer();
        let batch = proc.prepare_batch(&["a", "b", "c", "d"], &tok);
        let splits = BatchProcessor::split_batch(batch, 2);
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].batch_size, 2);
        assert_eq!(splits[1].batch_size, 2);
    }

    #[test]
    fn dynamic_sort_orders_by_length() {
        let texts = &["longer text", "hi", "medium"];
        let (sorted, _indices) = BatchProcessor::dynamic_batch_sort(texts);
        let lens: Vec<usize> = sorted.iter().map(|s| s.len()).collect();
        assert!(lens.windows(2).all(|w| w[0] <= w[1]));
    }
}
