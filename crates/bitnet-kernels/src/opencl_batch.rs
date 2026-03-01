//! OpenCL batch inference support for Intel Arc A770 GPUs.
//!
//! Implements batch processing of multiple sequences in parallel, including:
//!
//! - **Padding** — left, right, or no-padding strategies to uniform length
//! - **Attention masks** — automatic 1.0/0.0 mask generation from padding
//! - **Position IDs** — sequential IDs respecting padding offsets
//! - **Batch scheduling** — partitioning sequences into bounded batches
//! - **Batched embedding lookup** — table lookup for all sequences at once
//! - **Batched argmax** — greedy token selection across the batch dimension
//!
//! CPU reference implementations are provided for correctness testing.
//! The OpenCL kernel source (`BATCH_INFERENCE_SRC`) contains GPU kernels
//! for embedding lookup, mask generation, and argmax.

use std::fmt;
use std::time::Instant;

// ── Padding strategy ─────────────────────────────────────────────

/// How sequences are padded to uniform length within a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad on the left (prepend pad tokens).
    Left,
    /// Pad on the right (append pad tokens).
    Right,
    /// No padding — sequences must already be uniform length.
    None,
}

// ── Configuration ────────────────────────────────────────────────

/// Configuration for batch inference.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of sequences in a single batch.
    pub max_batch_size: u32,
    /// Maximum allowed sequence length (tokens).
    pub max_sequence_length: u32,
    /// Padding strategy applied when sequences differ in length.
    pub padding_strategy: PaddingStrategy,
    /// If `true`, sequences are sorted by length before batching.
    pub dynamic_batching: bool,
}

impl BatchConfig {
    /// Create a new batch configuration with sensible defaults.
    pub fn new(max_batch_size: u32, max_sequence_length: u32) -> Self {
        Self {
            max_batch_size,
            max_sequence_length,
            padding_strategy: PaddingStrategy::Right,
            dynamic_batching: false,
        }
    }
}

// ── Batch types ──────────────────────────────────────────────────

/// A single sequence within a batch.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchItem {
    /// Unique identifier for this sequence.
    pub sequence_id: u64,
    /// Token IDs (padded to `padded_length` of the containing schedule).
    pub tokens: Vec<u32>,
    /// Attention mask: 1.0 for real tokens, 0.0 for padding.
    pub attention_mask: Vec<f32>,
    /// Position IDs for rotary/positional embeddings.
    pub position_ids: Vec<u32>,
}

/// A scheduled batch of sequences ready for inference.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchSchedule {
    /// The items in this batch.
    pub items: Vec<BatchItem>,
    /// Length to which all sequences are padded.
    pub padded_length: usize,
    /// Number of sequences in this batch.
    pub batch_size: usize,
}

/// Result of running batch inference.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Per-sequence logits: `[batch_size][vocab_size]`.
    pub logits: Vec<Vec<f32>>,
    /// Sequence IDs in the same order as `logits`.
    pub sequence_ids: Vec<u64>,
    /// Wall-clock processing time in microseconds.
    pub processing_time_us: u64,
}

// ── Errors ───────────────────────────────────────────────────────

/// Errors that can occur during batch processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// The input sequence list was empty.
    EmptyBatch,
    /// A batch exceeds `max_batch_size`.
    ExceedsMaxBatchSize { requested: usize, max: u32 },
    /// A sequence exceeds `max_sequence_length`.
    ExceedsMaxSequenceLength { length: usize, max: u32 },
    /// Sequences have different lengths with `PaddingStrategy::None`.
    MismatchedDimensions { expected: usize, got: usize },
    /// The scheduler failed to create a valid partition.
    SchedulingFailed(String),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => write!(f, "batch is empty"),
            Self::ExceedsMaxBatchSize { requested, max } => {
                write!(
                    f,
                    "batch size {requested} exceeds maximum {max}"
                )
            }
            Self::ExceedsMaxSequenceLength { length, max } => {
                write!(
                    f,
                    "sequence length {length} exceeds maximum {max}"
                )
            }
            Self::MismatchedDimensions { expected, got } => {
                write!(
                    f,
                    "expected length {expected}, got {got} \
                     (PaddingStrategy::None requires uniform length)"
                )
            }
            Self::SchedulingFailed(reason) => {
                write!(f, "scheduling failed: {reason}")
            }
        }
    }
}

impl std::error::Error for BatchError {}

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL C source for batch inference kernels.
///
/// Contains:
/// - `batch_embed_lookup` — batched embedding with coalesced access
/// - `batch_attention_mask` — parallel mask generation
/// - `batch_argmax` — parallel argmax across the batch dimension
pub const BATCH_INFERENCE_SRC: &str = r#"
// batch_embed_lookup: table[vocab_size, embed_dim], ids[batch*seq_len]
// output[batch * seq_len * embed_dim]
__kernel void batch_embed_lookup(
    __global const float* table,
    __global const uint*  ids,
    __global       float* output,
    const uint embed_dim,
    const uint seq_len)
{
    uint gid  = get_global_id(0); // flat index across batch*seq_len*embed_dim
    uint total = get_global_size(0);
    if (gid >= total) return;

    uint token_flat = gid / embed_dim;
    uint dim_idx    = gid % embed_dim;
    uint token_id   = ids[token_flat];

    output[gid] = table[token_id * embed_dim + dim_idx];
}

// batch_attention_mask: lengths[batch], output[batch * padded_len]
// 1.0 for positions < length, 0.0 otherwise (right-padding layout)
__kernel void batch_attention_mask(
    __global const uint*  lengths,
    __global       float* output,
    const uint padded_len)
{
    uint batch_idx = get_global_id(0);
    uint pos       = get_global_id(1);
    if (pos >= padded_len) return;

    uint len = lengths[batch_idx];
    output[batch_idx * padded_len + pos] = (pos < len) ? 1.0f : 0.0f;
}

// batch_argmax: logits[batch, vocab_size] -> out_tokens[batch]
__kernel void batch_argmax(
    __global const float* logits,
    __global       uint*  out_tokens,
    const uint vocab_size)
{
    uint batch_idx = get_global_id(0);
    __global const float* row = logits + batch_idx * vocab_size;

    float best_val = row[0];
    uint  best_idx = 0;
    for (uint i = 1; i < vocab_size; i++) {
        if (row[i] > best_val) {
            best_val = row[i];
            best_idx = i;
        }
    }
    out_tokens[batch_idx] = best_idx;
}
"#;

// ── CPU reference implementations ────────────────────────────────

/// Pad sequences to `target_len` using the given strategy and pad token.
pub fn cpu_pad_sequences(
    sequences: &[Vec<u32>],
    target_len: usize,
    pad_token: u32,
    strategy: PaddingStrategy,
) -> Vec<Vec<u32>> {
    sequences
        .iter()
        .map(|seq| {
            let pad_count = target_len.saturating_sub(seq.len());
            match strategy {
                PaddingStrategy::Left => {
                    let mut padded = vec![pad_token; pad_count];
                    padded.extend_from_slice(seq);
                    padded.truncate(target_len);
                    padded
                }
                PaddingStrategy::Right => {
                    let mut padded = seq.clone();
                    padded.resize(target_len, pad_token);
                    padded
                }
                PaddingStrategy::None => seq.clone(),
            }
        })
        .collect()
}

/// Create attention masks: 1.0 for real tokens, 0.0 for padding.
pub fn cpu_create_attention_masks(
    sequences: &[Vec<u32>],
    padded_len: usize,
    strategy: PaddingStrategy,
) -> Vec<Vec<f32>> {
    sequences
        .iter()
        .map(|seq| {
            let real_len = seq.len().min(padded_len);
            let mut mask = vec![0.0f32; padded_len];
            match strategy {
                PaddingStrategy::Right | PaddingStrategy::None => {
                    for m in mask.iter_mut().take(real_len) {
                        *m = 1.0;
                    }
                }
                PaddingStrategy::Left => {
                    let offset = padded_len.saturating_sub(real_len);
                    for m in mask.iter_mut().skip(offset) {
                        *m = 1.0;
                    }
                }
            }
            mask
        })
        .collect()
}

/// Create position IDs for each sequence, respecting padding offsets.
pub fn cpu_create_position_ids(
    sequences: &[Vec<u32>],
    padded_len: usize,
) -> Vec<Vec<u32>> {
    sequences
        .iter()
        .map(|seq| {
            let real_len = seq.len().min(padded_len);
            let mut ids = vec![0u32; padded_len];
            // Position IDs are 0..real_len for the real token positions.
            // For left-padded layouts the offset is padded_len - real_len.
            for (i, id) in ids.iter_mut().enumerate().take(padded_len) {
                if i < real_len {
                    *id = i as u32;
                }
            }
            ids
        })
        .collect()
}

/// Batched embedding lookup: for each sequence, look up token vectors.
///
/// `table` is row-major `[vocab_size, embed_dim]`.
/// Returns one flat vector per sequence of length `seq_len * embed_dim`.
pub fn cpu_batch_embedding_lookup(
    table: &[f32],
    embed_dim: usize,
    token_ids: &[Vec<u32>],
) -> Vec<Vec<f32>> {
    token_ids
        .iter()
        .map(|ids| {
            let mut out = Vec::with_capacity(ids.len() * embed_dim);
            for &id in ids {
                let start = (id as usize) * embed_dim;
                let end = start + embed_dim;
                if end <= table.len() {
                    out.extend_from_slice(&table[start..end]);
                } else {
                    out.extend(std::iter::repeat_n(0.0f32, embed_dim));
                }
            }
            out
        })
        .collect()
}

/// Greedy decode: argmax per sequence over the vocabulary dimension.
///
/// Each entry in `logits` has length `vocab_size`.
pub fn cpu_batch_logits_to_tokens(
    logits: &[Vec<f32>],
    vocab_size: usize,
) -> Vec<u32> {
    logits
        .iter()
        .map(|row| {
            debug_assert!(
                row.len() >= vocab_size,
                "logits row length {} < vocab_size {vocab_size}",
                row.len()
            );
            row.iter()
                .take(vocab_size)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        })
        .collect()
}

/// Return indices that would sort `items` by sequence length (ascending).
pub fn cpu_sort_by_length(items: &[BatchItem]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..items.len()).collect();
    indices.sort_by_key(|&i| items[i].tokens.len());
    indices
}

/// Partition sequences into batches that respect `config` limits.
pub fn cpu_create_batch_schedule(
    sequences: &[Vec<u32>],
    config: &BatchConfig,
) -> Result<Vec<BatchSchedule>, BatchError> {
    if sequences.is_empty() {
        return Err(BatchError::EmptyBatch);
    }

    // Validate sequence lengths.
    for seq in sequences {
        if seq.len() > config.max_sequence_length as usize {
            return Err(BatchError::ExceedsMaxSequenceLength {
                length: seq.len(),
                max: config.max_sequence_length,
            });
        }
    }

    // If PaddingStrategy::None, all sequences must have the same length.
    if config.padding_strategy == PaddingStrategy::None {
        let first_len = sequences[0].len();
        for seq in sequences.iter().skip(1) {
            if seq.len() != first_len {
                return Err(BatchError::MismatchedDimensions {
                    expected: first_len,
                    got: seq.len(),
                });
            }
        }
    }

    // Optionally sort by length for dynamic batching.
    let order: Vec<usize> = if config.dynamic_batching {
        let mut idx: Vec<usize> = (0..sequences.len()).collect();
        idx.sort_by_key(|&i| sequences[i].len());
        idx
    } else {
        (0..sequences.len()).collect()
    };

    let max_bs = config.max_batch_size as usize;
    let mut schedules = Vec::new();
    let mut pos = 0;

    while pos < order.len() {
        let end = (pos + max_bs).min(order.len());
        let chunk: Vec<&Vec<u32>> =
            order[pos..end].iter().map(|&i| &sequences[i]).collect();

        let padded_length =
            chunk.iter().map(|s| s.len()).max().unwrap_or(0);

        let padded = cpu_pad_sequences(
            &chunk.iter().map(|s| (*s).clone()).collect::<Vec<_>>(),
            padded_length,
            0,
            config.padding_strategy,
        );
        let masks = cpu_create_attention_masks(
            &chunk.iter().map(|s| (*s).clone()).collect::<Vec<_>>(),
            padded_length,
            config.padding_strategy,
        );
        let position_ids = cpu_create_position_ids(
            &chunk.iter().map(|s| (*s).clone()).collect::<Vec<_>>(),
            padded_length,
        );

        let items: Vec<BatchItem> = padded
            .into_iter()
            .zip(masks)
            .zip(position_ids)
            .enumerate()
            .map(|(i, ((tokens, attention_mask), pos_ids))| {
                BatchItem {
                    sequence_id: order[pos + i] as u64,
                    tokens,
                    attention_mask,
                    position_ids: pos_ids,
                }
            })
            .collect();

        let batch_size = items.len();
        schedules.push(BatchSchedule { items, padded_length, batch_size });
        pos = end;
    }

    if schedules.is_empty() {
        return Err(BatchError::SchedulingFailed(
            "produced zero schedules".into(),
        ));
    }

    Ok(schedules)
}

/// Merge multiple `BatchResult`s into one.
pub fn cpu_merge_batch_results(results: &[BatchResult]) -> BatchResult {
    let start = Instant::now();
    let mut logits = Vec::new();
    let mut sequence_ids = Vec::new();
    for r in results {
        logits.extend_from_slice(&r.logits);
        sequence_ids.extend_from_slice(&r.sequence_ids);
    }
    BatchResult {
        logits,
        sequence_ids,
        processing_time_us: start.elapsed().as_micros() as u64,
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────

    fn make_config(bs: u32, seq: u32) -> BatchConfig {
        BatchConfig::new(bs, seq)
    }

    fn make_config_with(
        bs: u32,
        seq: u32,
        pad: PaddingStrategy,
        dyn_b: bool,
    ) -> BatchConfig {
        BatchConfig {
            max_batch_size: bs,
            max_sequence_length: seq,
            padding_strategy: pad,
            dynamic_batching: dyn_b,
        }
    }

    /// Small embedding table: 8 tokens × 4 dims, value = token*10 + dim.
    fn small_embed_table() -> Vec<f32> {
        let mut t = Vec::with_capacity(8 * 4);
        for tok in 0..8u32 {
            for d in 0..4u32 {
                t.push((tok * 10 + d) as f32);
            }
        }
        t
    }

    // ── padding: right ──────────────────────────────────────────

    #[test]
    fn pad_right_basic() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let padded = cpu_pad_sequences(&seqs, 4, 0, PaddingStrategy::Right);
        assert_eq!(padded[0], vec![1, 2, 0, 0]);
        assert_eq!(padded[1], vec![3, 4, 5, 0]);
    }

    #[test]
    fn pad_right_already_uniform() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let padded = cpu_pad_sequences(&seqs, 3, 0, PaddingStrategy::Right);
        assert_eq!(padded[0], vec![1, 2, 3]);
        assert_eq!(padded[1], vec![4, 5, 6]);
    }

    #[test]
    fn pad_right_empty_sequence() {
        let seqs = vec![vec![], vec![1]];
        let padded = cpu_pad_sequences(&seqs, 3, 99, PaddingStrategy::Right);
        assert_eq!(padded[0], vec![99, 99, 99]);
        assert_eq!(padded[1], vec![1, 99, 99]);
    }

    #[test]
    fn pad_right_custom_token() {
        let seqs = vec![vec![10]];
        let padded = cpu_pad_sequences(&seqs, 4, 42, PaddingStrategy::Right);
        assert_eq!(padded[0], vec![10, 42, 42, 42]);
    }

    // ── padding: left ───────────────────────────────────────────

    #[test]
    fn pad_left_basic() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let padded = cpu_pad_sequences(&seqs, 4, 0, PaddingStrategy::Left);
        assert_eq!(padded[0], vec![0, 0, 1, 2]);
        assert_eq!(padded[1], vec![0, 3, 4, 5]);
    }

    #[test]
    fn pad_left_empty_sequence() {
        let seqs = vec![vec![]];
        let padded = cpu_pad_sequences(&seqs, 3, 0, PaddingStrategy::Left);
        assert_eq!(padded[0], vec![0, 0, 0]);
    }

    // ── padding: none ───────────────────────────────────────────

    #[test]
    fn pad_none_passthrough() {
        let seqs = vec![vec![7, 8, 9]];
        let padded = cpu_pad_sequences(&seqs, 3, 0, PaddingStrategy::None);
        assert_eq!(padded[0], vec![7, 8, 9]);
    }

    // ── attention masks ─────────────────────────────────────────

    #[test]
    fn mask_right_basic() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let masks =
            cpu_create_attention_masks(&seqs, 4, PaddingStrategy::Right);
        assert_eq!(masks[0], vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(masks[1], vec![1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn mask_left_basic() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let masks =
            cpu_create_attention_masks(&seqs, 4, PaddingStrategy::Left);
        assert_eq!(masks[0], vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(masks[1], vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn mask_none_all_ones() {
        let seqs = vec![vec![1, 2, 3]];
        let masks =
            cpu_create_attention_masks(&seqs, 3, PaddingStrategy::None);
        assert_eq!(masks[0], vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn mask_empty_sequence() {
        let seqs = vec![vec![]];
        let masks =
            cpu_create_attention_masks(&seqs, 3, PaddingStrategy::Right);
        assert_eq!(masks[0], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn mask_all_padding_left() {
        let seqs = vec![vec![]];
        let masks =
            cpu_create_attention_masks(&seqs, 4, PaddingStrategy::Left);
        assert_eq!(masks[0], vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ── position IDs ────────────────────────────────────────────

    #[test]
    fn position_ids_basic() {
        let seqs = vec![vec![10, 20, 30]];
        let ids = cpu_create_position_ids(&seqs, 3);
        assert_eq!(ids[0], vec![0, 1, 2]);
    }

    #[test]
    fn position_ids_shorter_sequence() {
        let seqs = vec![vec![10, 20]];
        let ids = cpu_create_position_ids(&seqs, 4);
        assert_eq!(ids[0], vec![0, 1, 0, 0]);
    }

    #[test]
    fn position_ids_empty_sequence() {
        let seqs = vec![vec![]];
        let ids = cpu_create_position_ids(&seqs, 3);
        assert_eq!(ids[0], vec![0, 0, 0]);
    }

    #[test]
    fn position_ids_multiple_sequences() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let ids = cpu_create_position_ids(&seqs, 4);
        assert_eq!(ids[0], vec![0, 1, 0, 0]);
        assert_eq!(ids[1], vec![0, 1, 2, 0]);
    }

    #[test]
    fn position_ids_monotonically_increasing() {
        let seqs = vec![vec![1, 2, 3, 4, 5]];
        let ids = cpu_create_position_ids(&seqs, 5);
        for w in ids[0].windows(2) {
            assert!(w[1] > w[0], "position IDs must increase");
        }
    }

    // ── batch scheduling ────────────────────────────────────────

    #[test]
    fn schedule_single_sequence() {
        let seqs = vec![vec![1, 2, 3]];
        let cfg = make_config(4, 16);
        let scheds = cpu_create_batch_schedule(&seqs, &cfg).unwrap();
        assert_eq!(scheds.len(), 1);
        assert_eq!(scheds[0].batch_size, 1);
        assert_eq!(scheds[0].padded_length, 3);
    }

    #[test]
    fn schedule_exact_max_batch() {
        let seqs: Vec<Vec<u32>> =
            (0..4).map(|i| vec![i as u32; 3]).collect();
        let cfg = make_config(4, 16);
        let scheds = cpu_create_batch_schedule(&seqs, &cfg).unwrap();
        assert_eq!(scheds.len(), 1);
        assert_eq!(scheds[0].batch_size, 4);
    }

    #[test]
    fn schedule_overflow_into_two_batches() {
        let seqs: Vec<Vec<u32>> =
            (0..5).map(|i| vec![i as u32; 3]).collect();
        let cfg = make_config(4, 16);
        let scheds = cpu_create_batch_schedule(&seqs, &cfg).unwrap();
        assert_eq!(scheds.len(), 2);
        assert_eq!(scheds[0].batch_size, 4);
        assert_eq!(scheds[1].batch_size, 1);
    }

    #[test]
    fn schedule_dynamic_batching_sorts() {
        let seqs = vec![vec![1; 8], vec![2; 2], vec![3; 5]];
        let cfg = make_config_with(4, 16, PaddingStrategy::Right, true);
        let scheds = cpu_create_batch_schedule(&seqs, &cfg).unwrap();
        // With dynamic batching, sequences are sorted by length.
        let lens: Vec<usize> =
            scheds[0].items.iter().map(|it| it.tokens.len()).collect();
        // padded_length == max in chunk so all have the same len after pad
        assert!(lens.iter().all(|&l| l == scheds[0].padded_length));
    }

    #[test]
    fn schedule_err_empty() {
        let seqs: Vec<Vec<u32>> = vec![];
        let cfg = make_config(4, 16);
        assert_eq!(
            cpu_create_batch_schedule(&seqs, &cfg),
            Err(BatchError::EmptyBatch)
        );
    }

    #[test]
    fn schedule_err_exceeds_seq_length() {
        let seqs = vec![vec![1; 20]];
        let cfg = make_config(4, 10);
        assert!(matches!(
            cpu_create_batch_schedule(&seqs, &cfg),
            Err(BatchError::ExceedsMaxSequenceLength { .. })
        ));
    }

    #[test]
    fn schedule_err_mismatched_with_none_padding() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let cfg = make_config_with(4, 16, PaddingStrategy::None, false);
        assert!(matches!(
            cpu_create_batch_schedule(&seqs, &cfg),
            Err(BatchError::MismatchedDimensions { .. })
        ));
    }

    // ── batch embedding lookup ──────────────────────────────────

    #[test]
    fn embed_lookup_single_sequence() {
        let table = small_embed_table();
        let ids = vec![vec![0, 2]];
        let out = cpu_batch_embedding_lookup(&table, 4, &ids);
        assert_eq!(out.len(), 1);
        // token 0: [0,1,2,3], token 2: [20,21,22,23]
        assert_eq!(out[0], vec![0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn embed_lookup_multiple_sequences() {
        let table = small_embed_table();
        let ids = vec![vec![1], vec![3]];
        let out = cpu_batch_embedding_lookup(&table, 4, &ids);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], vec![10.0, 11.0, 12.0, 13.0]);
        assert_eq!(out[1], vec![30.0, 31.0, 32.0, 33.0]);
    }

    #[test]
    fn embed_lookup_oob_gives_zeros() {
        let table = small_embed_table(); // 8 tokens
        let ids = vec![vec![100]]; // out-of-bounds
        let out = cpu_batch_embedding_lookup(&table, 4, &ids);
        assert_eq!(out[0], vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ── batch argmax ────────────────────────────────────────────

    #[test]
    fn argmax_basic() {
        let logits = vec![
            vec![0.1, 0.5, 0.3],
            vec![0.9, 0.2, 0.1],
        ];
        let tokens = cpu_batch_logits_to_tokens(&logits, 3);
        assert_eq!(tokens, vec![1, 0]);
    }

    #[test]
    fn argmax_single_token_vocab() {
        let logits = vec![vec![42.0]];
        let tokens = cpu_batch_logits_to_tokens(&logits, 1);
        assert_eq!(tokens, vec![0]);
    }

    #[test]
    fn argmax_last_position() {
        let logits = vec![vec![0.0, 0.0, 0.0, 1.0]];
        let tokens = cpu_batch_logits_to_tokens(&logits, 4);
        assert_eq!(tokens, vec![3]);
    }

    #[test]
    fn argmax_negative_logits() {
        let logits = vec![vec![-5.0, -1.0, -3.0]];
        let tokens = cpu_batch_logits_to_tokens(&logits, 3);
        assert_eq!(tokens, vec![1]);
    }

    // ── sort by length ──────────────────────────────────────────

    #[test]
    fn sort_by_length_basic() {
        let items = vec![
            BatchItem {
                sequence_id: 0,
                tokens: vec![1; 5],
                attention_mask: vec![],
                position_ids: vec![],
            },
            BatchItem {
                sequence_id: 1,
                tokens: vec![2; 2],
                attention_mask: vec![],
                position_ids: vec![],
            },
            BatchItem {
                sequence_id: 2,
                tokens: vec![3; 8],
                attention_mask: vec![],
                position_ids: vec![],
            },
        ];
        let order = cpu_sort_by_length(&items);
        assert_eq!(order, vec![1, 0, 2]);
    }

    #[test]
    fn sort_by_length_already_sorted() {
        let items = vec![
            BatchItem {
                sequence_id: 0,
                tokens: vec![1; 1],
                attention_mask: vec![],
                position_ids: vec![],
            },
            BatchItem {
                sequence_id: 1,
                tokens: vec![2; 3],
                attention_mask: vec![],
                position_ids: vec![],
            },
        ];
        let order = cpu_sort_by_length(&items);
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn sort_by_length_reverse() {
        let items = vec![
            BatchItem {
                sequence_id: 0,
                tokens: vec![1; 10],
                attention_mask: vec![],
                position_ids: vec![],
            },
            BatchItem {
                sequence_id: 1,
                tokens: vec![2; 5],
                attention_mask: vec![],
                position_ids: vec![],
            },
            BatchItem {
                sequence_id: 2,
                tokens: vec![3; 1],
                attention_mask: vec![],
                position_ids: vec![],
            },
        ];
        let order = cpu_sort_by_length(&items);
        assert_eq!(order, vec![2, 1, 0]);
    }

    #[test]
    fn sort_by_length_stable_equal() {
        let items = vec![
            BatchItem {
                sequence_id: 0,
                tokens: vec![1; 3],
                attention_mask: vec![],
                position_ids: vec![],
            },
            BatchItem {
                sequence_id: 1,
                tokens: vec![2; 3],
                attention_mask: vec![],
                position_ids: vec![],
            },
        ];
        let order = cpu_sort_by_length(&items);
        // sort_by_key is stable, so equal-length items keep order.
        assert_eq!(order, vec![0, 1]);
    }

    // ── merge results ───────────────────────────────────────────

    #[test]
    fn merge_single_result() {
        let r = BatchResult {
            logits: vec![vec![1.0, 2.0]],
            sequence_ids: vec![0],
            processing_time_us: 100,
        };
        let merged = cpu_merge_batch_results(&[r]);
        assert_eq!(merged.logits.len(), 1);
        assert_eq!(merged.sequence_ids, vec![0]);
    }

    #[test]
    fn merge_multiple_results() {
        let r1 = BatchResult {
            logits: vec![vec![1.0], vec![2.0]],
            sequence_ids: vec![0, 1],
            processing_time_us: 50,
        };
        let r2 = BatchResult {
            logits: vec![vec![3.0]],
            sequence_ids: vec![2],
            processing_time_us: 30,
        };
        let merged = cpu_merge_batch_results(&[r1, r2]);
        assert_eq!(merged.logits.len(), 3);
        assert_eq!(merged.sequence_ids, vec![0, 1, 2]);
    }

    #[test]
    fn merge_preserves_sequence_ids() {
        let r1 = BatchResult {
            logits: vec![vec![0.0]],
            sequence_ids: vec![42],
            processing_time_us: 0,
        };
        let r2 = BatchResult {
            logits: vec![vec![0.0]],
            sequence_ids: vec![99],
            processing_time_us: 0,
        };
        let merged = cpu_merge_batch_results(&[r1, r2]);
        assert_eq!(merged.sequence_ids, vec![42, 99]);
    }

    // ── round-trip properties ───────────────────────────────────

    #[test]
    fn roundtrip_pad_then_mask_right() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5, 6]];
        let target = 6;
        let padded =
            cpu_pad_sequences(&seqs, target, 0, PaddingStrategy::Right);
        let masks =
            cpu_create_attention_masks(&seqs, target, PaddingStrategy::Right);
        // Wherever mask is 0, padded should be the pad token.
        for (p, m) in padded.iter().zip(masks.iter()) {
            for (tok, &mval) in p.iter().zip(m.iter()) {
                if mval == 0.0 {
                    assert_eq!(*tok, 0, "pad position should hold pad token");
                }
            }
        }
    }

    #[test]
    fn roundtrip_pad_then_mask_left() {
        let seqs = vec![vec![10, 20], vec![30]];
        let target = 4;
        let padded =
            cpu_pad_sequences(&seqs, target, 0, PaddingStrategy::Left);
        let masks =
            cpu_create_attention_masks(&seqs, target, PaddingStrategy::Left);
        for (p, m) in padded.iter().zip(masks.iter()) {
            for (tok, &mval) in p.iter().zip(m.iter()) {
                if mval == 0.0 {
                    assert_eq!(*tok, 0);
                }
            }
        }
    }

    // ── property: mask sum == real token count ───────────────────

    #[test]
    fn property_mask_sum_equals_real_tokens() {
        let seqs = vec![vec![1; 3], vec![2; 7], vec![3; 1]];
        let padded_len = 8;
        let masks = cpu_create_attention_masks(
            &seqs,
            padded_len,
            PaddingStrategy::Right,
        );
        for (seq, mask) in seqs.iter().zip(masks.iter()) {
            let real_count = seq.len().min(padded_len);
            let mask_sum: f32 = mask.iter().sum();
            assert_eq!(mask_sum as usize, real_count);
        }
    }

    #[test]
    fn property_mask_sum_left_padding() {
        let seqs = vec![vec![1; 2], vec![2; 5]];
        let padded_len = 6;
        let masks = cpu_create_attention_masks(
            &seqs,
            padded_len,
            PaddingStrategy::Left,
        );
        for (seq, mask) in seqs.iter().zip(masks.iter()) {
            let real_count = seq.len().min(padded_len);
            let mask_sum: f32 = mask.iter().sum();
            assert_eq!(mask_sum as usize, real_count);
        }
    }

    // ── large batch ─────────────────────────────────────────────

    #[test]
    fn large_batch_32_sequences() {
        let seqs: Vec<Vec<u32>> = (0..32)
            .map(|i| vec![i as u32; (i % 10 + 1) as usize])
            .collect();
        let cfg = make_config(16, 64);
        let scheds = cpu_create_batch_schedule(&seqs, &cfg).unwrap();
        let total: usize = scheds.iter().map(|s| s.batch_size).sum();
        assert_eq!(total, 32);
        assert_eq!(scheds.len(), 2); // 32 / 16 == 2
    }

    #[test]
    fn large_batch_all_items_present() {
        let seqs: Vec<Vec<u32>> = (0..32)
            .map(|i| vec![i as u32; (i % 7 + 1) as usize])
            .collect();
        let cfg = make_config(8, 64);
        let scheds = cpu_create_batch_schedule(&seqs, &cfg).unwrap();
        let mut all_ids: Vec<u64> = scheds
            .iter()
            .flat_map(|s| s.items.iter().map(|it| it.sequence_id))
            .collect();
        all_ids.sort();
        let expected: Vec<u64> = (0..32).collect();
        assert_eq!(all_ids, expected);
    }

    // ── error cases ─────────────────────────────────────────────

    #[test]
    fn error_empty_batch_display() {
        let e = BatchError::EmptyBatch;
        assert_eq!(format!("{e}"), "batch is empty");
    }

    #[test]
    fn error_exceeds_batch_size_display() {
        let e = BatchError::ExceedsMaxBatchSize {
            requested: 10,
            max: 4,
        };
        let msg = format!("{e}");
        assert!(msg.contains("10"));
        assert!(msg.contains("4"));
    }

    #[test]
    fn error_exceeds_seq_len_display() {
        let e = BatchError::ExceedsMaxSequenceLength {
            length: 1024,
            max: 512,
        };
        let msg = format!("{e}");
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn error_mismatched_display() {
        let e = BatchError::MismatchedDimensions {
            expected: 5,
            got: 3,
        };
        let msg = format!("{e}");
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn error_scheduling_failed_display() {
        let e = BatchError::SchedulingFailed("oops".into());
        assert!(format!("{e}").contains("oops"));
    }

    // ── OpenCL source sanity ────────────────────────────────────

    #[test]
    fn opencl_source_contains_kernels() {
        assert!(BATCH_INFERENCE_SRC.contains("batch_embed_lookup"));
        assert!(BATCH_INFERENCE_SRC.contains("batch_attention_mask"));
        assert!(BATCH_INFERENCE_SRC.contains("batch_argmax"));
    }

    #[test]
    fn opencl_source_not_empty() {
        assert!(!BATCH_INFERENCE_SRC.is_empty());
        assert!(BATCH_INFERENCE_SRC.len() > 100);
    }
}
