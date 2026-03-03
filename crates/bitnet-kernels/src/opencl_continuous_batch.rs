//! Continuous batching engine for OpenCL multi-request serving.
//!
//! Implements dynamic sequence scheduling inspired by vLLM/Orca,
//! enabling concurrent prefill and decode phases within a single batch.
//! Designed for Intel Arc A770 GPUs via OpenCL, with CPU reference
//! implementations for correctness testing.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for the continuous batching engine.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of concurrent sequences.
    pub max_batch_size: usize,
    /// Maximum sequence length (prompt + generated tokens).
    pub max_seq_len: usize,
    /// Number of tokens consumed per prefill iteration.
    pub prefill_chunk_size: usize,
    /// Whether priority-based preemption is enabled.
    pub enable_preemption: bool,
}

/// State of a sequence within the batch.
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceState {
    /// Processing prompt tokens.
    Prefilling,
    /// Auto-regressive token generation.
    Decoding,
    /// Temporarily paused (e.g. preempted).
    Paused,
    /// Generation finished normally.
    Completed,
    /// Generation failed with reason.
    Failed(String),
}

/// Metadata and state for a single in-flight sequence.
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    /// Unique identifier.
    pub id: u64,
    /// Original prompt token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Tokens produced so far.
    pub generated_tokens: Vec<u32>,
    /// Current lifecycle state.
    pub state: SequenceState,
    /// Scheduling priority (higher = more important).
    pub priority: u32,
    /// Arrival timestamp in nanoseconds.
    pub arrival_time_ns: u64,
}

/// A slot that may hold a sequence.
#[derive(Debug, Clone)]
pub struct BatchSlot {
    /// Index within the batch.
    pub slot_id: usize,
    /// Currently assigned sequence, if any.
    pub sequence: Option<SequenceInfo>,
    /// Whether KV-cache memory has been reserved.
    pub kv_allocated: bool,
}

/// The continuous batching engine.
#[derive(Debug)]
pub struct ContinuousBatch {
    /// Fixed-size slot array.
    pub slots: Vec<BatchSlot>,
    /// Engine configuration.
    pub config: BatchConfig,
    /// Monotonic iteration counter.
    pub iteration_count: u64,
    /// Cumulative statistics.
    pub stats: BatchStats,
    /// Counter for generating unique sequence IDs.
    next_seq_id: u64,
}

/// Aggregate statistics for the batch engine.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub total_sequences_served: u64,
    pub total_tokens_generated: u64,
    pub avg_prefill_latency_ms: f64,
    pub avg_decode_latency_ms: f64,
    pub preemptions: u64,
    pub utilization_pct: f64,
}

/// An action the scheduler may emit.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleAction {
    /// Begin or continue prefilling the given sequence.
    AddToPrefill(u64),
    /// Continue decoding the given sequence.
    ContinueDecode(u64),
    /// Preempt (pause) the given sequence.
    Preempt(u64),
    /// Evict the given sequence from the batch.
    Evict(u64),
    /// Nothing to do.
    NoOp,
}

/// Errors produced by the batching engine.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchError {
    /// All batch slots are occupied.
    BatchFull,
    /// No sequence with the given ID was found.
    SequenceNotFound(u64),
    /// The operation is not valid for the current sequence state.
    InvalidState,
    /// Preemption could not be carried out.
    PreemptionFailed,
    /// Configuration or parameter error.
    ConfigError(String),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchFull => write!(f, "batch is full"),
            Self::SequenceNotFound(id) => write!(f, "sequence {id} not found"),
            Self::InvalidState => write!(f, "invalid sequence state for operation"),
            Self::PreemptionFailed => write!(f, "preemption failed"),
            Self::ConfigError(msg) => write!(f, "config error: {msg}"),
        }
    }
}

impl std::error::Error for BatchError {}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Create a new continuous batch from the given configuration.
pub fn create_continuous_batch(config: BatchConfig) -> ContinuousBatch {
    let slots = (0..config.max_batch_size)
        .map(|i| BatchSlot { slot_id: i, sequence: None, kv_allocated: false })
        .collect();
    ContinuousBatch {
        slots,
        config,
        iteration_count: 0,
        stats: BatchStats::default(),
        next_seq_id: 1,
    }
}

/// Add a new sequence to the batch, returning its unique ID.
pub fn cpu_add_sequence(
    batch: &mut ContinuousBatch,
    prompt: Vec<u32>,
    priority: u32,
) -> Result<u64, BatchError> {
    if prompt.is_empty() {
        return Err(BatchError::ConfigError("prompt must not be empty".into()));
    }
    let slot =
        batch.slots.iter_mut().find(|s| s.sequence.is_none()).ok_or(BatchError::BatchFull)?;

    let id = batch.next_seq_id;
    batch.next_seq_id += 1;

    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    slot.sequence = Some(SequenceInfo {
        id,
        prompt_tokens: prompt,
        generated_tokens: Vec::new(),
        state: SequenceState::Prefilling,
        priority,
        arrival_time_ns: now_ns,
    });
    slot.kv_allocated = true;
    Ok(id)
}

/// Remove a sequence from the batch, returning its info.
pub fn cpu_remove_sequence(
    batch: &mut ContinuousBatch,
    seq_id: u64,
) -> Result<SequenceInfo, BatchError> {
    let slot = batch
        .slots
        .iter_mut()
        .find(|s| s.sequence.as_ref().is_some_and(|seq| seq.id == seq_id))
        .ok_or(BatchError::SequenceNotFound(seq_id))?;

    let info = slot.sequence.take().unwrap();
    slot.kv_allocated = false;
    batch.stats.total_sequences_served += 1;
    Ok(info)
}

/// Produce a list of scheduling actions for the current iteration.
///
/// Policy: prefilling sequences are scheduled first (lower latency-to-first-token),
/// followed by decoding sequences ordered by priority (descending).
pub fn cpu_schedule_iteration(batch: &mut ContinuousBatch) -> Vec<ScheduleAction> {
    batch.iteration_count += 1;

    let mut actions = Vec::new();

    // Collect prefilling sequences sorted by priority (descending).
    let mut prefilling: Vec<(u64, u32)> = batch
        .slots
        .iter()
        .filter_map(|s| {
            s.sequence
                .as_ref()
                .filter(|seq| seq.state == SequenceState::Prefilling)
                .map(|seq| (seq.id, seq.priority))
        })
        .collect();
    prefilling.sort_by(|a, b| b.1.cmp(&a.1));

    for (id, _) in &prefilling {
        actions.push(ScheduleAction::AddToPrefill(*id));
    }

    // Collect decoding sequences (sorted by priority desc).
    let mut decoding: Vec<(u64, u32)> = batch
        .slots
        .iter()
        .filter_map(|s| {
            s.sequence
                .as_ref()
                .filter(|seq| seq.state == SequenceState::Decoding)
                .map(|seq| (seq.id, seq.priority))
        })
        .collect();
    decoding.sort_by(|a, b| b.1.cmp(&a.1));

    for (id, _) in &decoding {
        actions.push(ScheduleAction::ContinueDecode(*id));
    }

    if actions.is_empty() {
        actions.push(ScheduleAction::NoOp);
    }

    actions
}

/// Execute a prefill step for the given sequence.
///
/// Consumes up to `prefill_chunk_size` prompt tokens. When the prompt is
/// exhausted the sequence transitions to `Decoding`. The returned token is
/// selected via argmax over `logits`.
pub fn cpu_execute_prefill_step(
    batch: &mut ContinuousBatch,
    seq_id: u64,
    logits: &[f32],
) -> Result<u32, BatchError> {
    let chunk_size = batch.config.prefill_chunk_size;
    let max_seq_len = batch.config.max_seq_len;
    let seq = batch
        .slots
        .iter_mut()
        .find_map(|s| {
            s.sequence
                .as_mut()
                .filter(|seq| seq.id == seq_id && seq.state == SequenceState::Prefilling)
        })
        .ok_or(BatchError::SequenceNotFound(seq_id))?;

    // Determine how many prompt tokens have been "processed".
    let processed = seq.generated_tokens.len() + chunk_size;
    let prompt_len = seq.prompt_tokens.len();

    let token = argmax(logits);

    // Check max sequence length.
    if seq.prompt_tokens.len() + seq.generated_tokens.len() + 1 > max_seq_len {
        seq.state = SequenceState::Completed;
        return Ok(token);
    }

    seq.generated_tokens.push(token);
    batch.stats.total_tokens_generated += 1;

    if processed >= prompt_len {
        seq.state = SequenceState::Decoding;
    }

    Ok(token)
}

/// Execute a single decode step for the given sequence.
///
/// The token is selected via argmax over `logits`.
pub fn cpu_execute_decode_step(
    batch: &mut ContinuousBatch,
    seq_id: u64,
    logits: &[f32],
) -> Result<u32, BatchError> {
    let max_seq_len = batch.config.max_seq_len;
    let seq = batch
        .slots
        .iter_mut()
        .find_map(|s| {
            s.sequence
                .as_mut()
                .filter(|seq| seq.id == seq_id && seq.state == SequenceState::Decoding)
        })
        .ok_or(BatchError::SequenceNotFound(seq_id))?;

    let token = argmax(logits);

    if seq.prompt_tokens.len() + seq.generated_tokens.len() + 1 > max_seq_len {
        seq.state = SequenceState::Completed;
        return Ok(token);
    }

    seq.generated_tokens.push(token);
    batch.stats.total_tokens_generated += 1;

    Ok(token)
}

/// Preempt (pause) a running sequence to free resources for higher-priority work.
pub fn cpu_preempt_sequence(batch: &mut ContinuousBatch, seq_id: u64) -> Result<(), BatchError> {
    if !batch.config.enable_preemption {
        return Err(BatchError::PreemptionFailed);
    }
    let seq = batch
        .slots
        .iter_mut()
        .find_map(|s| {
            s.sequence.as_mut().filter(|seq| {
                seq.id == seq_id
                    && (seq.state == SequenceState::Prefilling
                        || seq.state == SequenceState::Decoding)
            })
        })
        .ok_or(BatchError::SequenceNotFound(seq_id))?;

    seq.state = SequenceState::Paused;
    batch.stats.preemptions += 1;
    Ok(())
}

/// Resume a previously paused sequence.
pub fn cpu_resume_sequence(batch: &mut ContinuousBatch, seq_id: u64) -> Result<(), BatchError> {
    let seq = batch
        .slots
        .iter_mut()
        .find_map(|s| {
            s.sequence.as_mut().filter(|seq| seq.id == seq_id && seq.state == SequenceState::Paused)
        })
        .ok_or(BatchError::SequenceNotFound(seq_id))?;

    // If the prompt hasn't been fully consumed, resume in prefill mode.
    let prompt_len = seq.prompt_tokens.len();
    let chunk = batch.config.prefill_chunk_size;
    let processed = seq.generated_tokens.len() + chunk;
    if processed < prompt_len {
        seq.state = SequenceState::Prefilling;
    } else {
        seq.state = SequenceState::Decoding;
    }
    Ok(())
}

/// Return references to all active (non-empty) sequences.
pub fn cpu_get_active_sequences(batch: &ContinuousBatch) -> Vec<&SequenceInfo> {
    batch.slots.iter().filter_map(|s| s.sequence.as_ref()).collect()
}

/// Compute aggregate statistics for the batch.
pub fn cpu_compute_batch_stats(batch: &ContinuousBatch) -> BatchStats {
    let occupied = batch.slots.iter().filter(|s| s.sequence.is_some()).count();
    let utilization = if batch.config.max_batch_size > 0 {
        (occupied as f64 / batch.config.max_batch_size as f64) * 100.0
    } else {
        0.0
    };
    BatchStats {
        total_sequences_served: batch.stats.total_sequences_served,
        total_tokens_generated: batch.stats.total_tokens_generated,
        avg_prefill_latency_ms: batch.stats.avg_prefill_latency_ms,
        avg_decode_latency_ms: batch.stats.avg_decode_latency_ms,
        preemptions: batch.stats.preemptions,
        utilization_pct: utilization,
    }
}

/// Identify the lowest-priority active sequence whose priority is strictly
/// below `new_priority`, returning its ID as a preemption candidate.
pub fn cpu_should_preempt(batch: &ContinuousBatch, new_priority: u32) -> Option<u64> {
    if !batch.config.enable_preemption {
        return None;
    }
    batch
        .slots
        .iter()
        .filter_map(|s| {
            s.sequence.as_ref().filter(|seq| {
                (seq.state == SequenceState::Prefilling || seq.state == SequenceState::Decoding)
                    && seq.priority < new_priority
            })
        })
        .min_by_key(|seq| seq.priority)
        .map(|seq| seq.id)
}

/// Format a human-readable status string for the batch.
pub fn format_batch_status(batch: &ContinuousBatch) -> String {
    let active = cpu_get_active_sequences(batch);
    let prefilling = active.iter().filter(|s| s.state == SequenceState::Prefilling).count();
    let decoding = active.iter().filter(|s| s.state == SequenceState::Decoding).count();
    let paused = active.iter().filter(|s| s.state == SequenceState::Paused).count();
    let completed = active.iter().filter(|s| s.state == SequenceState::Completed).count();
    format!(
        "Batch(iter={}, slots={}/{}, prefill={}, decode={}, paused={}, completed={}, tokens={})",
        batch.iteration_count,
        active.len(),
        batch.config.max_batch_size,
        prefilling,
        decoding,
        paused,
        completed,
        batch.stats.total_tokens_generated,
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Argmax over a logit slice. Returns 0 for empty slices.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BatchConfig {
        BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 8,
            enable_preemption: true,
        }
    }

    fn simple_logits(best: usize, len: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        if best < len {
            v[best] = 1.0;
        }
        v
    }

    // ---- Creation ----------------------------------------------------------

    #[test]
    fn test_create_batch() {
        let batch = create_continuous_batch(default_config());
        assert_eq!(batch.slots.len(), 4);
        assert_eq!(batch.iteration_count, 0);
        assert!(batch.slots.iter().all(|s| s.sequence.is_none()));
    }

    #[test]
    fn test_create_batch_slot_ids() {
        let batch = create_continuous_batch(default_config());
        for (i, slot) in batch.slots.iter().enumerate() {
            assert_eq!(slot.slot_id, i);
            assert!(!slot.kv_allocated);
        }
    }

    // ---- Adding sequences --------------------------------------------------

    #[test]
    fn test_add_sequence() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1, 2, 3], 10).unwrap();
        assert_eq!(id, 1);
        let active = cpu_get_active_sequences(&batch);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].prompt_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_add_sequence_state_is_prefilling() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![10], 5).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Prefilling);
    }

    #[test]
    fn test_add_sequence_kv_allocated() {
        let mut batch = create_continuous_batch(default_config());
        cpu_add_sequence(&mut batch, vec![1], 1).unwrap();
        assert!(batch.slots[0].kv_allocated);
    }

    #[test]
    fn test_add_multiple_sequences() {
        let mut batch = create_continuous_batch(default_config());
        let id1 = cpu_add_sequence(&mut batch, vec![1], 1).unwrap();
        let id2 = cpu_add_sequence(&mut batch, vec![2], 2).unwrap();
        let id3 = cpu_add_sequence(&mut batch, vec![3], 3).unwrap();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_eq!(cpu_get_active_sequences(&batch).len(), 3);
    }

    #[test]
    fn test_add_sequence_unique_ids() {
        let mut batch = create_continuous_batch(default_config());
        let ids: Vec<u64> =
            (0..4).map(|_| cpu_add_sequence(&mut batch, vec![1], 0).unwrap()).collect();
        let set: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_add_sequence_empty_prompt_error() {
        let mut batch = create_continuous_batch(default_config());
        let err = cpu_add_sequence(&mut batch, vec![], 1).unwrap_err();
        assert!(matches!(err, BatchError::ConfigError(_)));
    }

    // ---- Batch full --------------------------------------------------------

    #[test]
    fn test_batch_full_error() {
        let mut batch = create_continuous_batch(default_config());
        for _ in 0..4 {
            cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        }
        let err = cpu_add_sequence(&mut batch, vec![1], 0).unwrap_err();
        assert_eq!(err, BatchError::BatchFull);
    }

    // ---- Removing sequences ------------------------------------------------

    #[test]
    fn test_remove_sequence() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1, 2], 5).unwrap();
        let info = cpu_remove_sequence(&mut batch, id).unwrap();
        assert_eq!(info.id, id);
        assert_eq!(cpu_get_active_sequences(&batch).len(), 0);
    }

    #[test]
    fn test_remove_frees_slot() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_remove_sequence(&mut batch, id).unwrap();
        assert!(!batch.slots[0].kv_allocated);
        // Should be able to add again
        cpu_add_sequence(&mut batch, vec![2], 0).unwrap();
    }

    #[test]
    fn test_remove_nonexistent_sequence() {
        let mut batch = create_continuous_batch(default_config());
        let err = cpu_remove_sequence(&mut batch, 999).unwrap_err();
        assert_eq!(err, BatchError::SequenceNotFound(999));
    }

    #[test]
    fn test_remove_increments_served_count() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_remove_sequence(&mut batch, id).unwrap();
        assert_eq!(batch.stats.total_sequences_served, 1);
    }

    // ---- Scheduling --------------------------------------------------------

    #[test]
    fn test_schedule_empty_batch() {
        let mut batch = create_continuous_batch(default_config());
        let actions = cpu_schedule_iteration(&mut batch);
        assert_eq!(actions, vec![ScheduleAction::NoOp]);
    }

    #[test]
    fn test_schedule_prefill_first() {
        let mut batch = create_continuous_batch(default_config());
        // Prompt of length 7 < chunk_size(8), so one prefill step transitions to Decoding
        let id1 = cpu_add_sequence(&mut batch, vec![1, 2, 3, 4, 5, 6, 7], 1).unwrap();

        // Transition id1: execute one prefill to move to Decoding
        cpu_execute_prefill_step(&mut batch, id1, &simple_logits(5, 10)).unwrap();
        // processed = 0 + 8 = 8 >= 7 -> Decoding

        let id2 =
            cpu_add_sequence(&mut batch, vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 2).unwrap();

        let actions = cpu_schedule_iteration(&mut batch);
        // Prefill should come before decode
        assert!(matches!(actions[0], ScheduleAction::AddToPrefill(_)));
        assert_eq!(actions[0], ScheduleAction::AddToPrefill(id2));
        assert!(actions.iter().any(|a| *a == ScheduleAction::ContinueDecode(id1)));
    }

    #[test]
    fn test_schedule_increments_iteration() {
        let mut batch = create_continuous_batch(default_config());
        cpu_schedule_iteration(&mut batch);
        cpu_schedule_iteration(&mut batch);
        assert_eq!(batch.iteration_count, 2);
    }

    #[test]
    fn test_schedule_priority_ordering_decode() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 1, // prompt consumed in one step
            enable_preemption: false,
        });
        let id_lo = cpu_add_sequence(&mut batch, vec![1], 1).unwrap();
        let id_hi = cpu_add_sequence(&mut batch, vec![2], 10).unwrap();

        // Move both to Decoding
        cpu_execute_prefill_step(&mut batch, id_lo, &simple_logits(0, 8)).unwrap();
        cpu_execute_prefill_step(&mut batch, id_hi, &simple_logits(0, 8)).unwrap();

        let actions = cpu_schedule_iteration(&mut batch);
        let decode_ids: Vec<u64> = actions
            .iter()
            .filter_map(|a| match a {
                ScheduleAction::ContinueDecode(id) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(decode_ids, vec![id_hi, id_lo]);
    }

    // ---- Prefill step ------------------------------------------------------

    #[test]
    fn test_prefill_generates_token() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1, 2], 0).unwrap();
        let tok = cpu_execute_prefill_step(&mut batch, id, &simple_logits(7, 10)).unwrap();
        assert_eq!(tok, 7);
    }

    #[test]
    fn test_prefill_transition_to_decoding() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: false,
        });
        let id = cpu_add_sequence(&mut batch, vec![1, 2, 3], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Decoding);
    }

    #[test]
    fn test_prefill_stays_prefilling_long_prompt() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 256,
            prefill_chunk_size: 2,
            enable_preemption: false,
        });
        let prompt: Vec<u32> = (0..20).collect();
        let id = cpu_add_sequence(&mut batch, prompt, 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Prefilling);
    }

    #[test]
    fn test_prefill_wrong_sequence_error() {
        let mut batch = create_continuous_batch(default_config());
        let err = cpu_execute_prefill_step(&mut batch, 999, &simple_logits(0, 8)).unwrap_err();
        assert_eq!(err, BatchError::SequenceNotFound(999));
    }

    #[test]
    fn test_prefill_updates_token_count() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(3, 8)).unwrap();
        assert_eq!(batch.stats.total_tokens_generated, 1);
    }

    // ---- Decode step -------------------------------------------------------

    #[test]
    fn test_decode_generates_token() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: false,
        });
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        let tok = cpu_execute_decode_step(&mut batch, id, &simple_logits(5, 8)).unwrap();
        assert_eq!(tok, 5);
    }

    #[test]
    fn test_decode_wrong_state_error() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0).unwrap();
        // Still prefilling – decode should fail to find it.
        let err = cpu_execute_decode_step(&mut batch, id, &simple_logits(0, 8)).unwrap_err();
        assert_eq!(err, BatchError::SequenceNotFound(id));
    }

    #[test]
    fn test_decode_increments_generated_tokens() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: false,
        });
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        cpu_execute_decode_step(&mut batch, id, &simple_logits(1, 8)).unwrap();
        cpu_execute_decode_step(&mut batch, id, &simple_logits(2, 8)).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.generated_tokens.len(), 3); // 1 from prefill + 2 from decode
    }

    #[test]
    fn test_decode_respects_max_seq_len() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 3, // prompt(1) + generated can be at most 3
            prefill_chunk_size: 4,
            enable_preemption: false,
        });
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap(); // gen=1, total=2
        cpu_execute_decode_step(&mut batch, id, &simple_logits(0, 8)).unwrap(); // gen=2, total=3
        // Next would exceed max_seq_len
        let tok = cpu_execute_decode_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Completed);
        assert_eq!(tok, 0);
    }

    // ---- Preemption --------------------------------------------------------

    #[test]
    fn test_preempt_sequence() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_preempt_sequence(&mut batch, id).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Paused);
    }

    #[test]
    fn test_preempt_increments_stat() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_preempt_sequence(&mut batch, id).unwrap();
        assert_eq!(batch.stats.preemptions, 1);
    }

    #[test]
    fn test_preempt_disabled() {
        let mut batch =
            create_continuous_batch(BatchConfig { enable_preemption: false, ..default_config() });
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        let err = cpu_preempt_sequence(&mut batch, id).unwrap_err();
        assert_eq!(err, BatchError::PreemptionFailed);
    }

    #[test]
    fn test_preempt_nonexistent() {
        let mut batch = create_continuous_batch(default_config());
        let err = cpu_preempt_sequence(&mut batch, 42).unwrap_err();
        assert_eq!(err, BatchError::SequenceNotFound(42));
    }

    #[test]
    fn test_preempt_already_paused() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_preempt_sequence(&mut batch, id).unwrap();
        let err = cpu_preempt_sequence(&mut batch, id).unwrap_err();
        assert_eq!(err, BatchError::SequenceNotFound(id));
    }

    // ---- Resume ------------------------------------------------------------

    #[test]
    fn test_resume_sequence() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: true,
        });
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        cpu_preempt_sequence(&mut batch, id).unwrap();
        cpu_resume_sequence(&mut batch, id).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Decoding);
    }

    #[test]
    fn test_resume_back_to_prefilling() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 256,
            prefill_chunk_size: 2,
            enable_preemption: true,
        });
        let prompt: Vec<u32> = (0..20).collect();
        let id = cpu_add_sequence(&mut batch, prompt, 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        cpu_preempt_sequence(&mut batch, id).unwrap();
        cpu_resume_sequence(&mut batch, id).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Prefilling);
    }

    #[test]
    fn test_resume_not_paused_error() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        let err = cpu_resume_sequence(&mut batch, id).unwrap_err();
        assert_eq!(err, BatchError::SequenceNotFound(id));
    }

    // ---- Should preempt ----------------------------------------------------

    #[test]
    fn test_should_preempt_finds_lowest() {
        let mut batch = create_continuous_batch(default_config());
        let id_low = cpu_add_sequence(&mut batch, vec![1], 1).unwrap();
        let _id_med = cpu_add_sequence(&mut batch, vec![2], 5).unwrap();
        let _id_hi = cpu_add_sequence(&mut batch, vec![3], 10).unwrap();
        let victim = cpu_should_preempt(&batch, 8);
        assert_eq!(victim, Some(id_low));
    }

    #[test]
    fn test_should_preempt_none_when_all_higher() {
        let mut batch = create_continuous_batch(default_config());
        cpu_add_sequence(&mut batch, vec![1], 10).unwrap();
        assert_eq!(cpu_should_preempt(&batch, 5), None);
    }

    #[test]
    fn test_should_preempt_disabled() {
        let mut batch =
            create_continuous_batch(BatchConfig { enable_preemption: false, ..default_config() });
        cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        assert_eq!(cpu_should_preempt(&batch, 100), None);
    }

    #[test]
    fn test_should_preempt_ignores_paused() {
        let mut batch = create_continuous_batch(default_config());
        let id = cpu_add_sequence(&mut batch, vec![1], 1).unwrap();
        cpu_preempt_sequence(&mut batch, id).unwrap();
        assert_eq!(cpu_should_preempt(&batch, 100), None);
    }

    // ---- Concurrent prefill + decode ---------------------------------------

    #[test]
    fn test_concurrent_prefill_and_decode() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 2,
            enable_preemption: false,
        });
        // Seq A: short prompt, will move to decode quickly
        let id_a = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id_a, &simple_logits(0, 8)).unwrap();
        // id_a should be Decoding now (prompt=1 < chunk=2 processed=1+2=3>=1)

        // Seq B: long prompt, still prefilling
        let _id_b =
            cpu_add_sequence(&mut batch, vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 0).unwrap();

        let actions = cpu_schedule_iteration(&mut batch);
        let has_prefill = actions.iter().any(|a| matches!(a, ScheduleAction::AddToPrefill(_)));
        let has_decode = actions.iter().any(|a| matches!(a, ScheduleAction::ContinueDecode(_)));
        assert!(has_prefill);
        assert!(has_decode);
    }

    // ---- Stats tracking ----------------------------------------------------

    #[test]
    fn test_stats_tokens_generated() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: false,
        });
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        cpu_execute_decode_step(&mut batch, id, &simple_logits(1, 8)).unwrap();
        cpu_execute_decode_step(&mut batch, id, &simple_logits(2, 8)).unwrap();
        assert_eq!(batch.stats.total_tokens_generated, 3);
    }

    #[test]
    fn test_compute_batch_stats_utilization() {
        let mut batch = create_continuous_batch(default_config());
        cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_add_sequence(&mut batch, vec![2], 0).unwrap();
        let stats = cpu_compute_batch_stats(&batch);
        assert!((stats.utilization_pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_batch_stats_empty() {
        let batch = create_continuous_batch(default_config());
        let stats = cpu_compute_batch_stats(&batch);
        assert!((stats.utilization_pct).abs() < f64::EPSILON);
        assert_eq!(stats.total_tokens_generated, 0);
    }

    // ---- Format status -----------------------------------------------------

    #[test]
    fn test_format_batch_status() {
        let mut batch = create_continuous_batch(default_config());
        cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        let status = format_batch_status(&batch);
        assert!(status.contains("slots=1/4"));
        assert!(status.contains("prefill=1"));
    }

    #[test]
    fn test_format_batch_status_empty() {
        let batch = create_continuous_batch(default_config());
        let status = format_batch_status(&batch);
        assert!(status.contains("slots=0/4"));
    }

    // ---- Edge: max_batch_size=1 --------------------------------------------

    #[test]
    fn test_single_slot_batch() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 1,
            max_seq_len: 64,
            prefill_chunk_size: 4,
            enable_preemption: true,
        });
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        assert_eq!(cpu_add_sequence(&mut batch, vec![2], 0).unwrap_err(), BatchError::BatchFull);
        cpu_remove_sequence(&mut batch, id).unwrap();
        cpu_add_sequence(&mut batch, vec![3], 0).unwrap();
    }

    // ---- Edge: all sequences completed -------------------------------------

    #[test]
    fn test_all_completed_schedule() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 2,
            max_seq_len: 2,
            prefill_chunk_size: 4,
            enable_preemption: false,
        });
        let id1 = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        let id2 = cpu_add_sequence(&mut batch, vec![2], 0).unwrap();

        // Force completion via max_seq_len
        cpu_execute_prefill_step(&mut batch, id1, &simple_logits(0, 8)).unwrap();
        cpu_execute_prefill_step(&mut batch, id2, &simple_logits(0, 8)).unwrap();
        // Now prompt(1)+generated(1)=2=max_seq_len; next decode triggers Completed
        cpu_execute_decode_step(&mut batch, id1, &simple_logits(0, 8)).unwrap();
        cpu_execute_decode_step(&mut batch, id2, &simple_logits(0, 8)).unwrap();

        let actions = cpu_schedule_iteration(&mut batch);
        assert_eq!(actions, vec![ScheduleAction::NoOp]);
    }

    // ---- Property: batch never exceeds max ---------------------------------

    #[test]
    fn test_property_batch_size_bounded() {
        let cfg = BatchConfig {
            max_batch_size: 3,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: false,
        };
        let mut batch = create_continuous_batch(cfg);
        for _ in 0..3 {
            cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        }
        assert!(cpu_add_sequence(&mut batch, vec![1], 0).is_err());
        assert!(cpu_get_active_sequences(&batch).len() <= 3);
    }

    // ---- Property: generated tokens monotonically increase -----------------

    #[test]
    fn test_property_generated_tokens_monotonic() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: false,
        });
        let id = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();

        let mut prev_len = 1usize; // from prefill
        for i in 0..10 {
            cpu_execute_decode_step(&mut batch, id, &simple_logits(i % 8, 8)).unwrap();
            let seq = batch
                .slots
                .iter()
                .find_map(|s| s.sequence.as_ref().filter(|s| s.id == id))
                .unwrap();
            assert!(seq.generated_tokens.len() > prev_len);
            prev_len = seq.generated_tokens.len();
        }
    }

    // ---- Argmax helper -----------------------------------------------------

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.9, 0.2]), 1);
    }

    #[test]
    fn test_argmax_empty() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn test_argmax_negative() {
        assert_eq!(argmax(&[-1.0, -0.5, -2.0]), 1);
    }

    // ---- BatchError display ------------------------------------------------

    #[test]
    fn test_batch_error_display() {
        assert_eq!(format!("{}", BatchError::BatchFull), "batch is full");
        assert!(format!("{}", BatchError::SequenceNotFound(7)).contains("7"));
        assert_eq!(format!("{}", BatchError::InvalidState), "invalid sequence state for operation");
    }

    // ---- Preempt then add higher priority ----------------------------------

    #[test]
    fn test_preempt_and_add_higher_priority() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 1,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: true,
        });
        let id_low = cpu_add_sequence(&mut batch, vec![1], 1).unwrap();
        assert_eq!(cpu_add_sequence(&mut batch, vec![2], 10).unwrap_err(), BatchError::BatchFull);

        // Preempt the low-priority sequence
        cpu_preempt_sequence(&mut batch, id_low).unwrap();
        // Slot is still occupied (paused), so still full
        assert_eq!(cpu_add_sequence(&mut batch, vec![2], 10).unwrap_err(), BatchError::BatchFull);

        // Remove the preempted sequence to truly free the slot
        cpu_remove_sequence(&mut batch, id_low).unwrap();
        let id_hi = cpu_add_sequence(&mut batch, vec![2], 10).unwrap();
        assert_ne!(id_hi, id_low);
    }

    // ---- Sequence state transitions ----------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut batch = create_continuous_batch(BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 4,
            enable_preemption: true,
        });
        let id = cpu_add_sequence(&mut batch, vec![1, 2], 5).unwrap();
        // Prefilling
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Prefilling);

        // Prefill -> Decoding
        cpu_execute_prefill_step(&mut batch, id, &simple_logits(0, 8)).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Decoding);

        // Decoding -> Paused
        cpu_preempt_sequence(&mut batch, id).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Paused);

        // Paused -> Decoding (resume)
        cpu_resume_sequence(&mut batch, id).unwrap();
        let seq =
            batch.slots.iter().find_map(|s| s.sequence.as_ref().filter(|s| s.id == id)).unwrap();
        assert_eq!(seq.state, SequenceState::Decoding);

        // Remove
        let info = cpu_remove_sequence(&mut batch, id).unwrap();
        assert_eq!(info.generated_tokens.len(), 1);
    }

    // ---- Multiple preemptions stats ----------------------------------------

    #[test]
    fn test_multiple_preemptions_tracked() {
        let mut batch = create_continuous_batch(default_config());
        let id1 = cpu_add_sequence(&mut batch, vec![1], 0).unwrap();
        let id2 = cpu_add_sequence(&mut batch, vec![2], 0).unwrap();
        cpu_preempt_sequence(&mut batch, id1).unwrap();
        cpu_preempt_sequence(&mut batch, id2).unwrap();
        assert_eq!(batch.stats.preemptions, 2);
    }

    // ---- SequenceState clone/eq --------------------------------------------

    #[test]
    fn test_sequence_state_eq() {
        assert_eq!(SequenceState::Prefilling, SequenceState::Prefilling);
        assert_ne!(SequenceState::Prefilling, SequenceState::Decoding);
        assert_eq!(SequenceState::Failed("x".into()), SequenceState::Failed("x".into()));
        assert_ne!(SequenceState::Failed("x".into()), SequenceState::Failed("y".into()));
    }
}
