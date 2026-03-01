//! Integration tests for GPU tokenizer and batch processor.

use bitnet_opencl::{
    BatchProcessor, GpuTokenizer, GpuTokenizerConfig, PaddingStrategy,
    TruncationStrategy,
};

// ---------------------------------------------------------------------------
// GpuTokenizer – encode
// ---------------------------------------------------------------------------

#[test]
fn single_text_encode_roundtrip() {
    let tok = GpuTokenizer::with_defaults();
    let ids = tok.encode("hello world");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, "hello world");
}

#[test]
fn batch_encode_correct_lengths() {
    let tok = GpuTokenizer::with_defaults();
    let batch = tok.encode_batch(&["ab", "cdef", "g"]);
    assert_eq!(batch.len(), 3);
    assert_eq!(batch[0].len(), 2);
    assert_eq!(batch[1].len(), 4);
    assert_eq!(batch[2].len(), 1);
}

#[test]
fn encode_empty_input_produces_empty_output() {
    let tok = GpuTokenizer::with_defaults();
    assert!(tok.encode("").is_empty());
}

#[test]
fn encode_batch_empty_slice() {
    let tok = GpuTokenizer::with_defaults();
    let batch = tok.encode_batch(&[]);
    assert!(batch.is_empty());
}

#[test]
fn encode_batch_with_empty_string() {
    let tok = GpuTokenizer::with_defaults();
    let batch = tok.encode_batch(&["", "a", ""]);
    assert!(batch[0].is_empty());
    assert_eq!(batch[1], vec![97]);
    assert!(batch[2].is_empty());
}

// ---------------------------------------------------------------------------
// GpuTokenizer – decode
// ---------------------------------------------------------------------------

#[test]
fn decode_simple_ascii() {
    let tok = GpuTokenizer::with_defaults();
    assert_eq!(tok.decode(&[72, 105]), "Hi");
}

#[test]
fn decode_empty_ids() {
    let tok = GpuTokenizer::with_defaults();
    assert_eq!(tok.decode(&[]), "");
}

#[test]
fn decode_skips_high_ids() {
    let tok = GpuTokenizer::with_defaults();
    assert_eq!(tok.decode(&[65, 999, 66]), "AB");
}

#[test]
fn decode_batch_matches_inputs() {
    let tok = GpuTokenizer::with_defaults();
    let ids1 = tok.encode("one");
    let ids2 = tok.encode("two");
    let decoded = tok.decode_batch(&[ids1.as_slice(), ids2.as_slice()]);
    assert_eq!(decoded, vec!["one", "two"]);
}

// ---------------------------------------------------------------------------
// Truncation
// ---------------------------------------------------------------------------

#[test]
fn max_length_truncation_left() {
    let cfg = GpuTokenizerConfig {
        max_length: 4,
        truncation: TruncationStrategy::TruncateLeft,
        ..Default::default()
    };
    let tok = GpuTokenizer::new(cfg);
    let ids = tok.encode("abcdef");
    assert_eq!(ids.len(), 4);
    assert_eq!(ids, vec![97, 98, 99, 100]);
}

#[test]
fn max_length_truncation_right() {
    let cfg = GpuTokenizerConfig {
        max_length: 4,
        truncation: TruncationStrategy::TruncateRight,
        ..Default::default()
    };
    let tok = GpuTokenizer::new(cfg);
    let ids = tok.encode("abcdef");
    assert_eq!(ids.len(), 4);
    assert_eq!(ids, vec![99, 100, 101, 102]);
}

#[test]
fn no_truncation_when_short() {
    let cfg = GpuTokenizerConfig { max_length: 100, ..Default::default() };
    let tok = GpuTokenizer::new(cfg);
    let ids = tok.encode("hi");
    assert_eq!(ids.len(), 2);
}

#[test]
fn truncation_none_does_not_truncate() {
    let cfg = GpuTokenizerConfig {
        max_length: 3,
        truncation: TruncationStrategy::None,
        ..Default::default()
    };
    let tok = GpuTokenizer::new(cfg);
    let ids = tok.encode("abcdef");
    assert_eq!(ids.len(), 6);
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

#[test]
fn padding_adds_correct_pad_tokens() {
    let cfg = GpuTokenizerConfig {
        padding: PaddingStrategy::Longest,
        max_length: 0,
        truncation: TruncationStrategy::None,
        ..Default::default()
    };
    let tok = GpuTokenizer::new(cfg);
    let seqs = vec![vec![1, 2, 3], vec![4, 5]];
    let (padded, lengths) = tok.pad_sequences(&seqs, 0);
    assert_eq!(lengths, vec![3, 2]);
    assert_eq!(padded[0], vec![1, 2, 3]);
    assert_eq!(padded[1], vec![4, 5, 0]);
}

#[test]
fn padding_max_length_strategy() {
    let cfg = GpuTokenizerConfig {
        padding: PaddingStrategy::MaxLength,
        max_length: 5,
        truncation: TruncationStrategy::None,
        ..Default::default()
    };
    let tok = GpuTokenizer::new(cfg);
    let seqs = vec![vec![1, 2]];
    let (padded, _) = tok.pad_sequences(&seqs, 99);
    assert_eq!(padded[0], vec![1, 2, 99, 99, 99]);
}

#[test]
fn padding_none_returns_originals() {
    let cfg = GpuTokenizerConfig { padding: PaddingStrategy::None, ..Default::default() };
    let tok = GpuTokenizer::new(cfg);
    let seqs = vec![vec![1, 2, 3], vec![4]];
    let (padded, lengths) = tok.pad_sequences(&seqs, 0);
    assert_eq!(padded, seqs);
    assert_eq!(lengths, vec![3, 1]);
}

#[test]
fn padding_empty_sequences() {
    let tok = GpuTokenizer::with_defaults();
    let (padded, lengths) = tok.pad_sequences(&[], 0);
    assert!(padded.is_empty());
    assert!(lengths.is_empty());
}

// ---------------------------------------------------------------------------
// Attention mask
// ---------------------------------------------------------------------------

#[test]
fn attention_mask_matches_padding() {
    let mask = GpuTokenizer::create_attention_mask(&[3, 2], 4);
    assert_eq!(mask[0], vec![1, 1, 1, 0]);
    assert_eq!(mask[1], vec![1, 1, 0, 0]);
}

#[test]
fn attention_mask_full_length() {
    let mask = GpuTokenizer::create_attention_mask(&[5], 5);
    assert_eq!(mask[0], vec![1, 1, 1, 1, 1]);
}

#[test]
fn attention_mask_zero_length() {
    let mask = GpuTokenizer::create_attention_mask(&[0], 3);
    assert_eq!(mask[0], vec![0, 0, 0]);
}

// ---------------------------------------------------------------------------
// BatchProcessor
// ---------------------------------------------------------------------------

#[test]
fn batch_processor_prepare_empty() {
    let proc = BatchProcessor::new(0);
    let tok = GpuTokenizer::with_defaults();
    let batch = proc.prepare_batch(&[], &tok);
    assert!(batch.is_empty());
}

#[test]
fn batch_processor_prepare_single() {
    let proc = BatchProcessor::new(0);
    let tok = GpuTokenizer::new(GpuTokenizerConfig { max_length: 128, ..Default::default() });
    let batch = proc.prepare_batch(&["ok"], &tok);
    assert_eq!(batch.batch_size, 1);
    assert_eq!(batch.input_ids[0], vec![111, 107]);
}

#[test]
fn batch_splitting_respects_max_size() {
    let proc = BatchProcessor::new(0);
    let tok = GpuTokenizer::with_defaults();
    let texts: Vec<&str> = (0..7).map(|_| "x").collect();
    let batch = proc.prepare_batch(texts.as_slice(), &tok);
    let splits = BatchProcessor::split_batch(batch, 3);
    assert_eq!(splits.len(), 3);
    assert_eq!(splits[0].batch_size, 3);
    assert_eq!(splits[1].batch_size, 3);
    assert_eq!(splits[2].batch_size, 1);
}

#[test]
fn split_batch_zero_max_returns_whole() {
    let proc = BatchProcessor::new(0);
    let tok = GpuTokenizer::with_defaults();
    let batch = proc.prepare_batch(&["a", "b"], &tok);
    let splits = BatchProcessor::split_batch(batch, 0);
    assert_eq!(splits.len(), 1);
}

// ---------------------------------------------------------------------------
// Dynamic batching
// ---------------------------------------------------------------------------

#[test]
fn dynamic_batching_groups_similar_lengths() {
    let texts = &["long text here", "hi", "medium"];
    let (sorted, indices) = BatchProcessor::dynamic_batch_sort(texts);
    // Sorted by ascending byte-length.
    let lens: Vec<usize> = sorted.iter().map(|s| s.len()).collect();
    assert!(lens.windows(2).all(|w| w[0] <= w[1]));
    // The index map can recover the original order.
    assert_eq!(indices.len(), texts.len());
}

#[test]
fn dynamic_batching_empty() {
    let (sorted, indices) = BatchProcessor::dynamic_batch_sort(&[]);
    assert!(sorted.is_empty());
    assert!(indices.is_empty());
}

#[test]
fn unsort_recovers_original_order() {
    let texts: &[&str] = &["cc", "a", "bbb"];
    let (sorted, indices) = BatchProcessor::dynamic_batch_sort(texts);
    let recovered = BatchProcessor::unsort(&sorted, &indices);
    assert_eq!(recovered, texts);
}

// ---------------------------------------------------------------------------
// InferenceBatch helpers
// ---------------------------------------------------------------------------

#[test]
fn inference_batch_max_seq_len() {
    let proc = BatchProcessor::new(0);
    let tok = GpuTokenizer::new(GpuTokenizerConfig {
        max_length: 128,
        padding: PaddingStrategy::Longest,
        ..Default::default()
    });
    let batch = proc.prepare_batch(&["abc", "de"], &tok);
    assert_eq!(batch.max_seq_len(), 3);
}

#[test]
fn inference_batch_positions_correct() {
    let proc = BatchProcessor::new(0);
    let tok = GpuTokenizer::new(GpuTokenizerConfig {
        max_length: 128,
        padding: PaddingStrategy::Longest,
        ..Default::default()
    });
    let batch = proc.prepare_batch(&["ab", "cdef"], &tok);
    // "ab" has length 2, padded to 4.
    assert_eq!(batch.positions[0], vec![0, 1, 0, 0]);
    assert_eq!(batch.positions[1], vec![0, 1, 2, 3]);
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[test]
fn default_config_values() {
    let cfg = GpuTokenizerConfig::default();
    assert_eq!(cfg.max_length, 512);
    assert_eq!(cfg.padding, PaddingStrategy::Longest);
    assert_eq!(cfg.truncation, TruncationStrategy::TruncateLeft);
}

#[test]
fn special_token_registration() {
    let mut tok = GpuTokenizer::with_defaults();
    tok.add_special_token("<pad>", 0);
    assert_eq!(tok.config().vocab_size, 50257);
}
