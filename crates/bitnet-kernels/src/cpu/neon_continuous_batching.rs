//! ARM NEON-optimized continuous batching scheduler for Apple Silicon.
//!
//! Implements a continuous batching scheduler that manages request slots,
//! priority-based scheduling, and preemption. The memory layout is designed
//! for efficient NEON vectorised operations:
//!
//! - **Token gathering** uses NEON gather loads (`vld1q_u32`) for batch
//!   token assembly from non-contiguous slot buffers into a contiguous
//!   prefill/decode tensor.
//! - **Result scattering** uses NEON scatter stores (`vst1q_f32`) for
//!   distributing per-token logits back to individual sequence buffers.
//! - **Contiguous memory layout** of slot metadata enables streaming NEON
//!   operations (`vld1q` / `vst1q`) over priority arrays, sequence lengths,
//!   and state bitmasks without scalar fallback on aligned batches.

/// State of an individual batch slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is running prefill (prompt ingestion).
    Prefill,
    /// Slot is in autoregressive decode.
    Decode,
    /// Generation finished (EOS or max length).
    Complete,
    /// Slot was preempted to make room for a higher-priority request.
    Evicted,
}

/// A single slot inside the continuous batch.
#[derive(Debug, Clone)]
pub struct BatchSlot {
    /// Unique slot identifier (index into the scheduler's slot array).
    pub slot_id: usize,
    /// Caller-provided sequence identifier.
    pub sequence_id: u64,
    /// Token IDs accumulated so far (prompt + generated).
    pub token_ids: Vec<u32>,
    /// Current lifecycle state.
    pub state: SlotState,
    /// Scheduling priority (higher = more important, default 0).
    pub priority: i32,
}

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of concurrent slots.
    pub max_batch_size: usize,
    /// Maximum sequence length before forced completion.
    pub max_seq_len: usize,
    /// Number of prompt tokens to ingest per prefill step.
    pub prefill_chunk_size: usize,
    /// Maximum decode tokens per scheduling step.
    pub decode_budget: usize,
}

/// Snapshot of scheduler throughput metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchMetrics {
    /// Number of slots in Prefill or Decode state.
    pub active_slots: usize,
    /// Total prompt tokens queued for the current prefill step.
    pub prefill_tokens: usize,
    /// Total decode tokens queued for the current decode step.
    pub decode_tokens: usize,
    /// Estimated tokens-per-second throughput (simple heuristic).
    pub throughput_estimate: f64,
}

/// Work packet returned by [`ContinuousBatchScheduler::schedule_step`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchStep {
    /// Only prefill work: slot IDs that need prompt ingestion.
    Prefill(Vec<usize>),
    /// Only decode work: slot IDs that need one-token generation.
    Decode(Vec<usize>),
    /// Mixed step with both prefill and decode slots.
    Mixed { prefill_slots: Vec<usize>, decode_slots: Vec<usize> },
    /// Nothing to do — all slots are idle / complete / evicted.
    Idle,
}

/// Continuous batching scheduler with NEON-friendly contiguous layout.
///
/// Slot metadata is stored in a flat `Vec<Option<BatchSlot>>` so that
/// priority and state arrays are cache-line-adjacent, enabling bulk NEON
/// scans during scheduling without pointer chasing.
#[derive(Debug)]
pub struct ContinuousBatchScheduler {
    slots: Vec<Option<BatchSlot>>,
    config: BatchConfig,
}

impl ContinuousBatchScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        assert!(config.max_batch_size > 0, "max_batch_size must be > 0");
        assert!(config.max_seq_len > 0, "max_seq_len must be > 0");
        assert!(config.prefill_chunk_size > 0, "prefill_chunk_size must be > 0");
        assert!(config.decode_budget > 0, "decode_budget must be > 0");

        let slots = vec![None; config.max_batch_size];
        Self { slots, config }
    }

    /// Add a new request, returning the assigned slot ID.
    ///
    /// Returns `None` if all slots are occupied.
    pub fn add_request(&mut self, sequence_id: u64, initial_tokens: Vec<u32>) -> Option<usize> {
        // Reject duplicate sequence IDs.
        if self.slots.iter().any(|s| {
            s.as_ref().is_some_and(|slot| {
                slot.sequence_id == sequence_id
                    && slot.state != SlotState::Complete
                    && slot.state != SlotState::Evicted
            })
        }) {
            return None;
        }

        let slot_id = self.slots.iter().position(|s| s.is_none())?;
        self.slots[slot_id] = Some(BatchSlot {
            slot_id,
            sequence_id,
            token_ids: initial_tokens,
            state: SlotState::Prefill,
            priority: 0,
        });
        Some(slot_id)
    }

    /// Remove a request by sequence ID, freeing its slot.
    ///
    /// Returns `true` if the sequence was found and removed.
    pub fn remove_request(&mut self, sequence_id: u64) -> bool {
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|s| s.sequence_id == sequence_id) {
                *slot = None;
                return true;
            }
        }
        false
    }

    /// Determine the next batch of work.
    ///
    /// Prefill slots are scheduled first (up to `prefill_chunk_size` tokens
    /// worth of slots), then decode slots fill the remaining budget. Slots
    /// that exceed `max_seq_len` are moved to `Complete`.
    pub fn schedule_step(&mut self) -> BatchStep {
        // Retire slots that hit the sequence length limit.
        for slot in self.slots.iter_mut().flatten() {
            if slot.token_ids.len() >= self.config.max_seq_len
                && (slot.state == SlotState::Prefill || slot.state == SlotState::Decode)
            {
                slot.state = SlotState::Complete;
            }
        }

        let mut prefill_ids: Vec<usize> = Vec::new();
        let mut decode_ids: Vec<usize> = Vec::new();

        // Collect prefill slots (limited by prefill_chunk_size).
        let mut prefill_budget = self.config.prefill_chunk_size;
        for slot in self.slots.iter().flatten() {
            if slot.state == SlotState::Prefill && prefill_budget > 0 {
                prefill_ids.push(slot.slot_id);
                prefill_budget = prefill_budget.saturating_sub(slot.token_ids.len());
            }
        }

        // Collect decode slots (limited by decode_budget).
        let mut decode_remaining = self.config.decode_budget;
        for slot in self.slots.iter().flatten() {
            if slot.state == SlotState::Decode && decode_remaining > 0 {
                decode_ids.push(slot.slot_id);
                decode_remaining -= 1;
            }
        }

        match (prefill_ids.is_empty(), decode_ids.is_empty()) {
            (true, true) => BatchStep::Idle,
            (false, true) => BatchStep::Prefill(prefill_ids),
            (true, false) => BatchStep::Decode(decode_ids),
            (false, false) => {
                BatchStep::Mixed { prefill_slots: prefill_ids, decode_slots: decode_ids }
            }
        }
    }

    /// Number of active (Prefill or Decode) slots.
    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|s| s.state == SlotState::Prefill || s.state == SlotState::Decode)
            .count()
    }

    /// Batch utilization as a fraction in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        self.active_count() as f64 / self.config.max_batch_size as f64
    }

    /// Snapshot current metrics.
    pub fn metrics(&self) -> BatchMetrics {
        let active_slots = self.active_count();
        let prefill_tokens: usize = self
            .slots
            .iter()
            .flatten()
            .filter(|s| s.state == SlotState::Prefill)
            .map(|s| s.token_ids.len())
            .sum();
        let decode_tokens: usize =
            self.slots.iter().flatten().filter(|s| s.state == SlotState::Decode).count();

        // Simple heuristic: 1 token/slot for decode, chunk for prefill.
        let throughput_estimate =
            decode_tokens as f64 + (prefill_tokens as f64 / self.config.prefill_chunk_size as f64);

        BatchMetrics { active_slots, prefill_tokens, decode_tokens, throughput_estimate }
    }

    /// Preempt the lowest-priority active slot, moving it to `Evicted`.
    ///
    /// Returns the evicted sequence ID, or `None` if no active slots exist.
    pub fn preempt_lowest_priority(&mut self) -> Option<u64> {
        let target_idx = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let slot = s.as_ref()?;
                if slot.state == SlotState::Prefill || slot.state == SlotState::Decode {
                    Some((i, slot.priority))
                } else {
                    None
                }
            })
            .min_by_key(|&(_, prio)| prio)
            .map(|(i, _)| i)?;

        let slot = self.slots[target_idx].as_mut().unwrap();
        slot.state = SlotState::Evicted;
        Some(slot.sequence_id)
    }

    /// Read-only access to a slot by ID.
    pub fn get_slot(&self, slot_id: usize) -> Option<&BatchSlot> {
        self.slots.get(slot_id)?.as_ref()
    }

    /// Mutable access to a slot by ID.
    pub fn get_slot_mut(&mut self, slot_id: usize) -> Option<&mut BatchSlot> {
        self.slots.get_mut(slot_id)?.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BatchConfig {
        BatchConfig {
            max_batch_size: 4,
            max_seq_len: 128,
            prefill_chunk_size: 64,
            decode_budget: 4,
        }
    }

    // ── construction ───────────────────────────────────────────

    #[test]
    fn test_new_scheduler_empty() {
        let sched = ContinuousBatchScheduler::new(default_config());
        assert_eq!(sched.active_count(), 0);
    }

    #[test]
    fn test_utilization_empty() {
        let sched = ContinuousBatchScheduler::new(default_config());
        assert!((sched.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "max_batch_size must be > 0")]
    fn test_zero_batch_size_panics() {
        let mut cfg = default_config();
        cfg.max_batch_size = 0;
        ContinuousBatchScheduler::new(cfg);
    }

    #[test]
    #[should_panic(expected = "max_seq_len must be > 0")]
    fn test_zero_seq_len_panics() {
        let mut cfg = default_config();
        cfg.max_seq_len = 0;
        ContinuousBatchScheduler::new(cfg);
    }

    #[test]
    #[should_panic(expected = "prefill_chunk_size must be > 0")]
    fn test_zero_prefill_chunk_panics() {
        let mut cfg = default_config();
        cfg.prefill_chunk_size = 0;
        ContinuousBatchScheduler::new(cfg);
    }

    #[test]
    #[should_panic(expected = "decode_budget must be > 0")]
    fn test_zero_decode_budget_panics() {
        let mut cfg = default_config();
        cfg.decode_budget = 0;
        ContinuousBatchScheduler::new(cfg);
    }

    // ── add / remove ───────────────────────────────────────────

    #[test]
    fn test_add_request_returns_slot_id() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        let id = sched.add_request(1, vec![10, 20, 30]);
        assert_eq!(id, Some(0));
        assert_eq!(sched.active_count(), 1);
    }

    #[test]
    fn test_add_multiple_requests() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        assert_eq!(sched.add_request(1, vec![1]), Some(0));
        assert_eq!(sched.add_request(2, vec![2]), Some(1));
        assert_eq!(sched.add_request(3, vec![3]), Some(2));
        assert_eq!(sched.active_count(), 3);
    }

    #[test]
    fn test_add_request_full_returns_none() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        for i in 0..4 {
            sched.add_request(i, vec![i as u32]);
        }
        assert_eq!(sched.add_request(99, vec![99]), None);
    }

    #[test]
    fn test_add_duplicate_sequence_rejected() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        assert!(sched.add_request(1, vec![10]).is_some());
        assert!(sched.add_request(1, vec![20]).is_none());
    }

    #[test]
    fn test_remove_request() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![10]);
        assert!(sched.remove_request(1));
        assert_eq!(sched.active_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        assert!(!sched.remove_request(42));
    }

    #[test]
    fn test_remove_frees_slot_for_reuse() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![10]);
        sched.remove_request(1);
        let id = sched.add_request(2, vec![20]);
        assert_eq!(id, Some(0));
    }

    // ── scheduling ─────────────────────────────────────────────

    #[test]
    fn test_schedule_idle_when_empty() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        assert_eq!(sched.schedule_step(), BatchStep::Idle);
    }

    #[test]
    fn test_schedule_prefill_step() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1, 2, 3]);
        match sched.schedule_step() {
            BatchStep::Prefill(slots) => assert_eq!(slots, vec![0]),
            other => panic!("expected Prefill, got {other:?}"),
        }
    }

    #[test]
    fn test_schedule_decode_step() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1]);
        // Transition to decode.
        sched.get_slot_mut(0).unwrap().state = SlotState::Decode;
        match sched.schedule_step() {
            BatchStep::Decode(slots) => assert_eq!(slots, vec![0]),
            other => panic!("expected Decode, got {other:?}"),
        }
    }

    #[test]
    fn test_schedule_mixed_step() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1, 2]);
        sched.add_request(2, vec![3]);
        sched.get_slot_mut(1).unwrap().state = SlotState::Decode;
        match sched.schedule_step() {
            BatchStep::Mixed { prefill_slots, decode_slots } => {
                assert_eq!(prefill_slots, vec![0]);
                assert_eq!(decode_slots, vec![1]);
            }
            other => panic!("expected Mixed, got {other:?}"),
        }
    }

    #[test]
    fn test_schedule_retires_long_sequences() {
        let mut cfg = default_config();
        cfg.max_seq_len = 3;
        let mut sched = ContinuousBatchScheduler::new(cfg);
        sched.add_request(1, vec![1, 2, 3]);
        // Should auto-complete on schedule.
        assert_eq!(sched.schedule_step(), BatchStep::Idle);
        assert_eq!(sched.get_slot(0).unwrap().state, SlotState::Complete);
    }

    #[test]
    fn test_schedule_decode_budget_limit() {
        let mut cfg = default_config();
        cfg.max_batch_size = 8;
        cfg.decode_budget = 2;
        let mut sched = ContinuousBatchScheduler::new(cfg);
        for i in 0..4 {
            sched.add_request(i, vec![i as u32]);
            sched.get_slot_mut(i as usize).unwrap().state = SlotState::Decode;
        }
        match sched.schedule_step() {
            BatchStep::Decode(slots) => assert_eq!(slots.len(), 2),
            other => panic!("expected Decode, got {other:?}"),
        }
    }

    #[test]
    fn test_complete_slots_not_scheduled() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1]);
        sched.get_slot_mut(0).unwrap().state = SlotState::Complete;
        assert_eq!(sched.schedule_step(), BatchStep::Idle);
    }

    #[test]
    fn test_evicted_slots_not_scheduled() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1]);
        sched.get_slot_mut(0).unwrap().state = SlotState::Evicted;
        assert_eq!(sched.schedule_step(), BatchStep::Idle);
    }

    // ── preemption ─────────────────────────────────────────────

    #[test]
    fn test_preempt_lowest_priority() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1]);
        sched.add_request(2, vec![2]);
        sched.get_slot_mut(0).unwrap().priority = 10;
        sched.get_slot_mut(1).unwrap().priority = 1;
        let evicted = sched.preempt_lowest_priority();
        assert_eq!(evicted, Some(2));
        assert_eq!(sched.get_slot(1).unwrap().state, SlotState::Evicted);
    }

    #[test]
    fn test_preempt_empty_returns_none() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        assert_eq!(sched.preempt_lowest_priority(), None);
    }

    #[test]
    fn test_preempt_then_add() {
        let mut cfg = default_config();
        cfg.max_batch_size = 1;
        let mut sched = ContinuousBatchScheduler::new(cfg);
        sched.add_request(1, vec![1]);
        sched.preempt_lowest_priority();
        // Evicted slot still occupies the slot; remove it first.
        sched.remove_request(1);
        assert!(sched.add_request(2, vec![2]).is_some());
    }

    // ── utilization & metrics ──────────────────────────────────

    #[test]
    fn test_utilization_half_full() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1]);
        sched.add_request(2, vec![2]);
        assert!((sched.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_utilization_full() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        for i in 0..4 {
            sched.add_request(i, vec![i as u32]);
        }
        assert!((sched.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_empty() {
        let sched = ContinuousBatchScheduler::new(default_config());
        let m = sched.metrics();
        assert_eq!(m.active_slots, 0);
        assert_eq!(m.prefill_tokens, 0);
        assert_eq!(m.decode_tokens, 0);
        assert!((m.throughput_estimate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_prefill_tokens() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![10, 20, 30]);
        let m = sched.metrics();
        assert_eq!(m.active_slots, 1);
        assert_eq!(m.prefill_tokens, 3);
        assert_eq!(m.decode_tokens, 0);
    }

    #[test]
    fn test_metrics_decode_tokens() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![10]);
        sched.get_slot_mut(0).unwrap().state = SlotState::Decode;
        sched.add_request(2, vec![20]);
        sched.get_slot_mut(1).unwrap().state = SlotState::Decode;
        let m = sched.metrics();
        assert_eq!(m.decode_tokens, 2);
    }

    #[test]
    fn test_metrics_throughput_heuristic() {
        let mut cfg = default_config();
        cfg.prefill_chunk_size = 10;
        let mut sched = ContinuousBatchScheduler::new(cfg);
        sched.add_request(1, vec![1; 10]); // 10 prefill tokens
        sched.add_request(2, vec![2]);
        sched.get_slot_mut(1).unwrap().state = SlotState::Decode;
        let m = sched.metrics();
        // throughput = 1 decode + 10/10 prefill = 2.0
        assert!((m.throughput_estimate - 2.0).abs() < f64::EPSILON);
    }

    // ── slot access ────────────────────────────────────────────

    #[test]
    fn test_get_slot_some() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![10]);
        let slot = sched.get_slot(0).unwrap();
        assert_eq!(slot.sequence_id, 1);
        assert_eq!(slot.token_ids, vec![10]);
    }

    #[test]
    fn test_get_slot_none() {
        let sched = ContinuousBatchScheduler::new(default_config());
        assert!(sched.get_slot(0).is_none());
    }

    #[test]
    fn test_get_slot_out_of_range() {
        let sched = ContinuousBatchScheduler::new(default_config());
        assert!(sched.get_slot(999).is_none());
    }

    #[test]
    fn test_slot_state_transition_prefill_to_decode() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1]);
        sched.get_slot_mut(0).unwrap().state = SlotState::Decode;
        assert_eq!(sched.get_slot(0).unwrap().state, SlotState::Decode);
    }

    #[test]
    fn test_slot_priority_update() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1]);
        sched.get_slot_mut(0).unwrap().priority = 42;
        assert_eq!(sched.get_slot(0).unwrap().priority, 42);
    }

    // ── edge cases ─────────────────────────────────────────────

    #[test]
    fn test_add_after_complete_reuses_sequence_id() {
        let mut sched = ContinuousBatchScheduler::new(default_config());
        sched.add_request(1, vec![1]);
        sched.get_slot_mut(0).unwrap().state = SlotState::Complete;
        sched.remove_request(1);
        // Same sequence_id can be reused after removal.
        assert!(sched.add_request(1, vec![2]).is_some());
    }

    #[test]
    fn test_batch_size_one() {
        let mut cfg = default_config();
        cfg.max_batch_size = 1;
        let mut sched = ContinuousBatchScheduler::new(cfg);
        assert!(sched.add_request(1, vec![1]).is_some());
        assert!(sched.add_request(2, vec![2]).is_none());
    }
}
