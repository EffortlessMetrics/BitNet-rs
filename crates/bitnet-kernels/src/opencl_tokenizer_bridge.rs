//! Tokenizer-GPU bridge for efficient token transfer between CPU and GPU buffers.
//!
//! Provides CPU reference implementations for preparing tokenized input into
//! GPU-friendly flat buffers, and for decoding generated token IDs back to text.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during bridge operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// A sequence exceeds the configured maximum length.
    ExceedsMaxLength { length: usize, max: usize },
    /// The input batch is empty.
    EmptyBatch,
    /// A token ID is not valid for the vocabulary.
    InvalidTokenId(u32),
    /// A token ID overflows the vocabulary size.
    VocabOverflow,
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExceedsMaxLength { length, max } => {
                write!(f, "sequence length {length} exceeds max {max}")
            }
            Self::EmptyBatch => write!(f, "empty batch"),
            Self::InvalidTokenId(id) => write!(f, "invalid token id: {id}"),
            Self::VocabOverflow => write!(f, "token id exceeds vocabulary size"),
        }
    }
}

impl std::error::Error for BridgeError {}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Strategy for padding sequences in a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad all sequences to the given length.
    MaxLength(usize),
    /// Pad all sequences to the length of the longest in the batch.
    LongestInBatch,
    /// No padding — sequences keep their original lengths.
    NoPadding,
}

/// Configuration for the tokenizer bridge.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub vocab_size: u32,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub max_length: usize,
}

/// A padded batch of token sequences ready for model consumption.
#[derive(Debug, Clone)]
pub struct TokenBatch {
    pub token_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub position_ids: Vec<Vec<u32>>,
    pub batch_size: usize,
    pub max_seq_len: usize,
}

/// Row-major flattened representation suitable for GPU transfer.
#[derive(Debug, Clone)]
pub struct GpuTokenBuffer {
    pub flat_tokens: Vec<u32>,
    pub flat_mask: Vec<u32>,
    pub flat_positions: Vec<u32>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Result of decoding token IDs back to text.
#[derive(Debug, Clone)]
pub struct DecodedOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub token_count: usize,
    pub special_tokens_removed: usize,
}

/// Cumulative statistics tracked by the bridge.
#[derive(Debug, Clone, Default)]
pub struct BridgeStats {
    pub batches_prepared: u64,
    pub total_tokens_transferred: u64,
    pub total_padding_tokens: u64,
    pub avg_seq_len: f64,
}

/// Main bridge struct that holds configuration and running stats.
#[derive(Debug, Clone)]
pub struct TokenBridge {
    pub config: TokenizerConfig,
    pub stats: BridgeStats,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Create a new [`TokenBridge`] with the given configuration.
pub fn create_token_bridge(config: TokenizerConfig) -> TokenBridge {
    TokenBridge { config, stats: BridgeStats::default() }
}

// ---------------------------------------------------------------------------
// Batch preparation
// ---------------------------------------------------------------------------

/// Prepare a padded [`TokenBatch`] from raw token sequences.
pub fn cpu_prepare_batch(
    bridge: &mut TokenBridge,
    sequences: Vec<Vec<u32>>,
    padding: PaddingStrategy,
) -> Result<TokenBatch, BridgeError> {
    if sequences.is_empty() {
        return Err(BridgeError::EmptyBatch);
    }

    // Validate lengths against max_length
    for seq in &sequences {
        if seq.len() > bridge.config.max_length {
            return Err(BridgeError::ExceedsMaxLength {
                length: seq.len(),
                max: bridge.config.max_length,
            });
        }
    }

    let target_len = match &padding {
        PaddingStrategy::MaxLength(len) => *len,
        PaddingStrategy::LongestInBatch => sequences.iter().map(|s| s.len()).max().unwrap_or(0),
        PaddingStrategy::NoPadding => sequences.iter().map(|s| s.len()).max().unwrap_or(0),
    };

    let padded = match &padding {
        PaddingStrategy::NoPadding => sequences.clone(),
        _ => cpu_pad_sequences(&sequences, target_len, bridge.config.pad_token_id),
    };

    let attention_mask =
        cpu_create_attention_mask(&sequences, target_len, bridge.config.pad_token_id);
    let position_ids = cpu_create_position_ids(&sequences, target_len);

    let batch_size = padded.len();

    // Update stats
    let real_tokens: u64 = sequences.iter().map(|s| s.len() as u64).sum();
    let total_slots = (batch_size as u64) * (target_len as u64);
    let padding_tokens = total_slots.saturating_sub(real_tokens);

    bridge.stats.batches_prepared += 1;
    bridge.stats.total_tokens_transferred += real_tokens;
    bridge.stats.total_padding_tokens += padding_tokens;

    let total_seqs =
        bridge.stats.total_tokens_transferred + bridge.stats.total_padding_tokens;
    if total_seqs > 0 {
        bridge.stats.avg_seq_len = bridge.stats.total_tokens_transferred as f64
            / bridge.stats.batches_prepared as f64
            / batch_size as f64;
    }

    Ok(TokenBatch {
        token_ids: padded,
        attention_mask,
        position_ids,
        batch_size,
        max_seq_len: target_len,
    })
}

// ---------------------------------------------------------------------------
// Padding helpers
// ---------------------------------------------------------------------------

/// Pad each sequence to `target_len` using `pad_id`.
pub fn cpu_pad_sequences(
    sequences: &[Vec<u32>],
    target_len: usize,
    pad_id: u32,
) -> Vec<Vec<u32>> {
    sequences
        .iter()
        .map(|seq| {
            let mut padded = seq.clone();
            padded.resize(target_len, pad_id);
            padded
        })
        .collect()
}

/// Create attention masks: 1 for real tokens, 0 for padding positions.
pub fn cpu_create_attention_mask(
    sequences: &[Vec<u32>],
    padded_len: usize,
    _pad_id: u32,
) -> Vec<Vec<u32>> {
    sequences
        .iter()
        .map(|seq| {
            let mut mask = vec![1u32; seq.len().min(padded_len)];
            mask.resize(padded_len, 0);
            mask
        })
        .collect()
}

/// Create position IDs: 0..real_len for real tokens, 0 for padding.
pub fn cpu_create_position_ids(
    sequences: &[Vec<u32>],
    padded_len: usize,
) -> Vec<Vec<u32>> {
    sequences
        .iter()
        .map(|seq| {
            let real_len = seq.len().min(padded_len);
            let mut positions: Vec<u32> = (0..real_len as u32).collect();
            positions.resize(padded_len, 0);
            positions
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Flatten / unflatten
// ---------------------------------------------------------------------------

/// Flatten a [`TokenBatch`] into a row-major [`GpuTokenBuffer`].
pub fn cpu_flatten_for_gpu(batch: &TokenBatch) -> GpuTokenBuffer {
    let seq_len = batch.max_seq_len;
    let mut flat_tokens = Vec::with_capacity(batch.batch_size * seq_len);
    let mut flat_mask = Vec::with_capacity(batch.batch_size * seq_len);
    let mut flat_positions = Vec::with_capacity(batch.batch_size * seq_len);

    for i in 0..batch.batch_size {
        flat_tokens.extend_from_slice(&batch.token_ids[i]);
        flat_mask.extend_from_slice(&batch.attention_mask[i]);
        flat_positions.extend_from_slice(&batch.position_ids[i]);
    }

    GpuTokenBuffer {
        flat_tokens,
        flat_mask,
        flat_positions,
        batch_size: batch.batch_size,
        seq_len,
    }
}

/// Restore 2D token ID vectors from a flat [`GpuTokenBuffer`].
pub fn cpu_unflatten_from_gpu(buffer: &GpuTokenBuffer) -> Vec<Vec<u32>> {
    buffer
        .flat_tokens
        .chunks(buffer.seq_len)
        .map(|chunk| chunk.to_vec())
        .collect()
}

// ---------------------------------------------------------------------------
// Decode / post-process
// ---------------------------------------------------------------------------

/// Decode token IDs to text using a vocabulary lookup table.
pub fn cpu_decode_token_ids(token_ids: &[u32], vocab: &[String]) -> DecodedOutput {
    let mut text = String::new();
    let mut valid_ids = Vec::new();
    for &id in token_ids {
        if let Some(tok) = vocab.get(id as usize) {
            text.push_str(tok);
            valid_ids.push(id);
        }
    }
    DecodedOutput {
        token_count: valid_ids.len(),
        text,
        token_ids: valid_ids,
        special_tokens_removed: 0,
    }
}

/// Remove special token IDs from a sequence.
pub fn cpu_remove_special_tokens(token_ids: &[u32], special_ids: &[u32]) -> Vec<u32> {
    token_ids.iter().copied().filter(|id| !special_ids.contains(id)).collect()
}

/// Truncate a sequence at the first occurrence of `eos_id`.
pub fn cpu_truncate_to_eos(token_ids: &[u32], eos_id: u32) -> Vec<u32> {
    if let Some(pos) = token_ids.iter().position(|&id| id == eos_id) {
        token_ids[..pos].to_vec()
    } else {
        token_ids.to_vec()
    }
}

/// Validate that all token IDs are within the vocabulary range.
pub fn cpu_validate_token_ids(
    token_ids: &[u32],
    vocab_size: u32,
) -> Result<(), BridgeError> {
    for &id in token_ids {
        if id >= vocab_size {
            return Err(BridgeError::InvalidTokenId(id));
        }
    }
    Ok(())
}

/// Return a snapshot of the current bridge statistics.
pub fn cpu_get_stats(bridge: &TokenBridge) -> BridgeStats {
    bridge.stats.clone()
}

/// Format a human-readable summary of a [`TokenBatch`].
pub fn format_batch_info(batch: &TokenBatch) -> String {
    let total_real: usize =
        batch.attention_mask.iter().map(|m| m.iter().map(|&v| v as usize).sum::<usize>()).sum();
    format!(
        "TokenBatch(batch_size={}, max_seq_len={}, real_tokens={}, padding_tokens={})",
        batch.batch_size,
        batch.max_seq_len,
        total_real,
        batch.batch_size * batch.max_seq_len - total_real,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TokenizerConfig {
        TokenizerConfig {
            vocab_size: 32000,
            pad_token_id: 0,
            eos_token_id: 2,
            bos_token_id: 1,
            max_length: 512,
        }
    }

    // --- Batch preparation ---

    #[test]
    fn test_prepare_batch_single_sequence() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 5, 10]],
            PaddingStrategy::NoPadding,
        )
        .unwrap();
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.max_seq_len, 3);
        assert_eq!(batch.token_ids[0], vec![1, 5, 10]);
    }

    #[test]
    fn test_prepare_batch_multiple_sequences_different_lengths() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3], vec![4, 5], vec![6]],
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        assert_eq!(batch.batch_size, 3);
        assert_eq!(batch.max_seq_len, 3);
        assert_eq!(batch.token_ids[1], vec![4, 5, 0]);
        assert_eq!(batch.token_ids[2], vec![6, 0, 0]);
    }

    #[test]
    fn test_prepare_batch_exceeds_max_length() {
        let mut bridge = create_token_bridge(TokenizerConfig {
            max_length: 3,
            ..default_config()
        });
        let err = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3, 4]],
            PaddingStrategy::NoPadding,
        )
        .unwrap_err();
        assert_eq!(err, BridgeError::ExceedsMaxLength { length: 4, max: 3 });
    }

    #[test]
    fn test_prepare_batch_empty_batch() {
        let mut bridge = create_token_bridge(default_config());
        let err =
            cpu_prepare_batch(&mut bridge, vec![], PaddingStrategy::NoPadding)
                .unwrap_err();
        assert_eq!(err, BridgeError::EmptyBatch);
    }

    // --- Padding ---

    #[test]
    fn test_padding_max_length() {
        let padded = cpu_pad_sequences(&[vec![1, 2], vec![3]], 5, 0);
        assert_eq!(padded[0], vec![1, 2, 0, 0, 0]);
        assert_eq!(padded[1], vec![3, 0, 0, 0, 0]);
    }

    #[test]
    fn test_padding_longest_in_batch() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3, 4], vec![5, 6]],
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        assert_eq!(batch.max_seq_len, 4);
        assert_eq!(batch.token_ids[1], vec![5, 6, 0, 0]);
    }

    #[test]
    fn test_padding_no_padding_preserves_lengths() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3]],
            PaddingStrategy::NoPadding,
        )
        .unwrap();
        assert_eq!(batch.token_ids[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_padding_already_correct_length() {
        let padded = cpu_pad_sequences(&[vec![1, 2, 3]], 3, 0);
        assert_eq!(padded[0], vec![1, 2, 3]);
    }

    // --- Attention mask ---

    #[test]
    fn test_attention_mask_real_vs_padding() {
        let mask = cpu_create_attention_mask(&[vec![1, 2], vec![3]], 4, 0);
        assert_eq!(mask[0], vec![1, 1, 0, 0]);
        assert_eq!(mask[1], vec![1, 0, 0, 0]);
    }

    #[test]
    fn test_attention_mask_no_padding() {
        let mask = cpu_create_attention_mask(&[vec![1, 2, 3]], 3, 0);
        assert_eq!(mask[0], vec![1, 1, 1]);
    }

    #[test]
    fn test_attention_mask_all_padding() {
        let mask = cpu_create_attention_mask(&[vec![]], 3, 0);
        assert_eq!(mask[0], vec![0, 0, 0]);
    }

    // --- Position IDs ---

    #[test]
    fn test_position_ids_sequential() {
        let pos = cpu_create_position_ids(&[vec![10, 20, 30]], 3);
        assert_eq!(pos[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_position_ids_with_padding() {
        let pos = cpu_create_position_ids(&[vec![10, 20]], 5);
        assert_eq!(pos[0], vec![0, 1, 0, 0, 0]);
    }

    #[test]
    fn test_position_ids_multiple_sequences() {
        let pos = cpu_create_position_ids(&[vec![1, 2, 3], vec![4]], 3);
        assert_eq!(pos[0], vec![0, 1, 2]);
        assert_eq!(pos[1], vec![0, 0, 0]);
    }

    // --- Flatten / unflatten ---

    #[test]
    fn test_flatten_unflatten_round_trip() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3], vec![4, 5, 6]],
            PaddingStrategy::NoPadding,
        )
        .unwrap();
        let buffer = cpu_flatten_for_gpu(&batch);
        let restored = cpu_unflatten_from_gpu(&buffer);
        assert_eq!(restored, batch.token_ids);
    }

    #[test]
    fn test_flatten_row_major_order() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2], vec![3, 4]],
            PaddingStrategy::NoPadding,
        )
        .unwrap();
        let buffer = cpu_flatten_for_gpu(&batch);
        assert_eq!(buffer.flat_tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_flatten_with_padding() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3], vec![4]],
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        let buffer = cpu_flatten_for_gpu(&batch);
        assert_eq!(buffer.flat_tokens, vec![1, 2, 3, 4, 0, 0]);
        assert_eq!(buffer.flat_mask, vec![1, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn test_unflatten_dimensions() {
        let buffer = GpuTokenBuffer {
            flat_tokens: vec![1, 2, 3, 4, 5, 6],
            flat_mask: vec![1, 1, 1, 1, 1, 1],
            flat_positions: vec![0, 1, 2, 0, 1, 2],
            batch_size: 2,
            seq_len: 3,
        };
        let restored = cpu_unflatten_from_gpu(&buffer);
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0], vec![1, 2, 3]);
        assert_eq!(restored[1], vec![4, 5, 6]);
    }

    // --- Decode ---

    #[test]
    fn test_decode_correct_text() {
        let vocab: Vec<String> =
            vec!["<pad>", "hello", " ", "world"].iter().map(|s| s.to_string()).collect();
        let output = cpu_decode_token_ids(&[1, 2, 3], &vocab);
        assert_eq!(output.text, "hello world");
        assert_eq!(output.token_count, 3);
    }

    #[test]
    fn test_decode_empty_ids() {
        let vocab: Vec<String> = vec!["a".to_string()];
        let output = cpu_decode_token_ids(&[], &vocab);
        assert_eq!(output.text, "");
        assert_eq!(output.token_count, 0);
    }

    #[test]
    fn test_decode_skips_out_of_range() {
        let vocab: Vec<String> = vec!["a".to_string(), "b".to_string()];
        let output = cpu_decode_token_ids(&[0, 99, 1], &vocab);
        assert_eq!(output.text, "ab");
        assert_eq!(output.token_count, 2);
    }

    // --- Remove special tokens ---

    #[test]
    fn test_remove_special_tokens_filters() {
        let result = cpu_remove_special_tokens(&[1, 5, 2, 10, 0], &[0, 1, 2]);
        assert_eq!(result, vec![5, 10]);
    }

    #[test]
    fn test_remove_special_tokens_none_present() {
        let result = cpu_remove_special_tokens(&[5, 10, 20], &[0, 1, 2]);
        assert_eq!(result, vec![5, 10, 20]);
    }

    #[test]
    fn test_remove_special_tokens_all_special() {
        let result = cpu_remove_special_tokens(&[0, 1, 2], &[0, 1, 2]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_remove_special_tokens_empty_input() {
        let result = cpu_remove_special_tokens(&[], &[0, 1, 2]);
        assert!(result.is_empty());
    }

    // --- Truncate at EOS ---

    #[test]
    fn test_truncate_at_eos() {
        let result = cpu_truncate_to_eos(&[1, 5, 2, 10, 20], 2);
        assert_eq!(result, vec![1, 5]);
    }

    #[test]
    fn test_truncate_no_eos_present() {
        let result = cpu_truncate_to_eos(&[1, 5, 10], 2);
        assert_eq!(result, vec![1, 5, 10]);
    }

    #[test]
    fn test_truncate_eos_at_start() {
        let result = cpu_truncate_to_eos(&[2, 5, 10], 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_truncate_eos_at_end() {
        let result = cpu_truncate_to_eos(&[1, 5, 2], 2);
        assert_eq!(result, vec![1, 5]);
    }

    // --- Validate ---

    #[test]
    fn test_validate_valid_ids() {
        assert!(cpu_validate_token_ids(&[0, 1, 99], 100).is_ok());
    }

    #[test]
    fn test_validate_invalid_id() {
        let err = cpu_validate_token_ids(&[0, 100], 100).unwrap_err();
        assert_eq!(err, BridgeError::InvalidTokenId(100));
    }

    #[test]
    fn test_validate_empty_is_ok() {
        assert!(cpu_validate_token_ids(&[], 100).is_ok());
    }

    #[test]
    fn test_validate_boundary_id() {
        assert!(cpu_validate_token_ids(&[99], 100).is_ok());
        assert!(cpu_validate_token_ids(&[100], 100).is_err());
    }

    // --- Edge cases ---

    #[test]
    fn test_single_token_sequence() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![42]],
            PaddingStrategy::NoPadding,
        )
        .unwrap();
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.max_seq_len, 1);
        assert_eq!(batch.token_ids[0], vec![42]);
        assert_eq!(batch.attention_mask[0], vec![1]);
        assert_eq!(batch.position_ids[0], vec![0]);
    }

    #[test]
    fn test_all_same_length_no_padding_needed() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2], vec![3, 4], vec![5, 6]],
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        // All have length 2, so no padding tokens added
        for mask in &batch.attention_mask {
            assert_eq!(mask, &vec![1, 1]);
        }
    }

    #[test]
    fn test_vocab_size_one() {
        assert!(cpu_validate_token_ids(&[0], 1).is_ok());
        assert!(cpu_validate_token_ids(&[1], 1).is_err());
    }

    // --- Property-based tests ---

    #[test]
    fn test_property_attention_mask_sum_equals_real_token_count() {
        let sequences = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
        let padded_len = 3;
        let mask = cpu_create_attention_mask(&sequences, padded_len, 0);
        for (seq, m) in sequences.iter().zip(mask.iter()) {
            let ones: u32 = m.iter().sum();
            assert_eq!(ones as usize, seq.len());
        }
    }

    #[test]
    fn test_property_position_ids_max_equals_seq_len_minus_one() {
        let sequences = vec![vec![10, 20, 30], vec![40, 50]];
        let padded_len = 5;
        let pos = cpu_create_position_ids(&sequences, padded_len);
        for (seq, p) in sequences.iter().zip(pos.iter()) {
            let max_pos = *p.iter().max().unwrap();
            assert_eq!(max_pos as usize, seq.len() - 1);
        }
    }

    #[test]
    fn test_property_flatten_preserves_total_elements() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3], vec![4, 5, 6]],
            PaddingStrategy::NoPadding,
        )
        .unwrap();
        let buffer = cpu_flatten_for_gpu(&batch);
        assert_eq!(buffer.flat_tokens.len(), batch.batch_size * batch.max_seq_len);
    }

    // --- Stats ---

    #[test]
    fn test_stats_batches_prepared() {
        let mut bridge = create_token_bridge(default_config());
        cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3]],
            PaddingStrategy::NoPadding,
        )
        .unwrap();
        cpu_prepare_batch(
            &mut bridge,
            vec![vec![4, 5]],
            PaddingStrategy::NoPadding,
        )
        .unwrap();
        let stats = cpu_get_stats(&bridge);
        assert_eq!(stats.batches_prepared, 2);
    }

    #[test]
    fn test_stats_total_tokens() {
        let mut bridge = create_token_bridge(default_config());
        cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3], vec![4, 5]],
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        let stats = cpu_get_stats(&bridge);
        assert_eq!(stats.total_tokens_transferred, 5);
    }

    #[test]
    fn test_stats_padding_tokens() {
        let mut bridge = create_token_bridge(default_config());
        cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3], vec![4]],
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        let stats = cpu_get_stats(&bridge);
        // 2 sequences × 3 slots = 6 total, 4 real → 2 padding
        assert_eq!(stats.total_padding_tokens, 2);
    }

    // --- Format ---

    #[test]
    fn test_format_batch_info() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3], vec![4]],
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        let info = format_batch_info(&batch);
        assert!(info.contains("batch_size=2"));
        assert!(info.contains("max_seq_len=3"));
        assert!(info.contains("real_tokens=4"));
        assert!(info.contains("padding_tokens=2"));
    }

    // --- BridgeError Display ---

    #[test]
    fn test_error_display_exceeds_max_length() {
        let e = BridgeError::ExceedsMaxLength { length: 10, max: 5 };
        assert!(e.to_string().contains("10"));
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_error_display_empty_batch() {
        assert_eq!(BridgeError::EmptyBatch.to_string(), "empty batch");
    }

    #[test]
    fn test_error_display_invalid_token_id() {
        let e = BridgeError::InvalidTokenId(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_error_display_vocab_overflow() {
        assert!(BridgeError::VocabOverflow.to_string().contains("vocab"));
    }

    // --- MaxLength padding strategy ---

    #[test]
    fn test_padding_strategy_max_length_pads_beyond_longest() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2]],
            PaddingStrategy::MaxLength(5),
        )
        .unwrap();
        assert_eq!(batch.max_seq_len, 5);
        assert_eq!(batch.token_ids[0], vec![1, 2, 0, 0, 0]);
    }

    // --- Round-trip with padding ---

    #[test]
    fn test_flatten_unflatten_with_padding_round_trip() {
        let mut bridge = create_token_bridge(default_config());
        let batch = cpu_prepare_batch(
            &mut bridge,
            vec![vec![1, 2, 3], vec![4]],
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        let buffer = cpu_flatten_for_gpu(&batch);
        let restored = cpu_unflatten_from_gpu(&buffer);
        assert_eq!(restored, batch.token_ids);
    }

    // --- Decode with special token removal ---

    #[test]
    fn test_decode_after_special_token_removal() {
        let vocab: Vec<String> =
            vec!["<pad>", "<bos>", "<eos>", "hi", " ", "there"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let ids = vec![1, 3, 4, 5, 2];
        let cleaned = cpu_remove_special_tokens(&ids, &[0, 1, 2]);
        let output = cpu_decode_token_ids(&cleaned, &vocab);
        assert_eq!(output.text, "hi there");
    }

    // --- Truncate + decode pipeline ---

    #[test]
    fn test_truncate_then_decode() {
        let vocab: Vec<String> =
            vec!["a", "b", "c", "<eos>"].iter().map(|s| s.to_string()).collect();
        let ids = vec![0, 1, 2, 3, 0, 1];
        let truncated = cpu_truncate_to_eos(&ids, 3);
        let output = cpu_decode_token_ids(&truncated, &vocab);
        assert_eq!(output.text, "abc");
    }

    // --- Large batch ---

    #[test]
    fn test_large_batch() {
        let mut bridge = create_token_bridge(default_config());
        let sequences: Vec<Vec<u32>> = (0..64).map(|i| vec![i; (i as usize % 10) + 1]).collect();
        let batch = cpu_prepare_batch(
            &mut bridge,
            sequences,
            PaddingStrategy::LongestInBatch,
        )
        .unwrap();
        assert_eq!(batch.batch_size, 64);
        assert_eq!(batch.max_seq_len, 10);
    }
}
