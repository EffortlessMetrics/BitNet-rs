//! OpenCL continuous (iteration-level) batching for serving multiple requests
//! with different generation lengths.
//!
//! Each forward-pass iteration processes one token per active slot.  Finished
//! requests free their slots immediately and new requests fill the gaps
//! without stalling the batch.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the continuous batching engine.
#[derive(Debug, Clone)]
pub struct ContinuousBatchConfig {
    /// Maximum number of concurrent generation slots.
    pub max_batch_size: usize,
    /// Maximum sequence length (prompt + generated tokens).
    pub max_seq_len: usize,
    /// Timeout in milliseconds for a single iteration.
    pub iteration_timeout_ms: u64,
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A request submitted for generation.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Unique request identifier (assigned by [`SlotManager`]).
    pub request_id: u64,
    /// Input token IDs (the prompt).
    pub prompt_tokens: Vec<u32>,
    /// Maximum number of *new* tokens to generate.
    pub max_tokens: usize,
    /// Scheduling priority (higher = more important).
    pub priority: u32,
}

/// A slot in the batch that may hold an active generation.
#[derive(Debug, Clone)]
pub struct GenerationSlot {
    /// Index within the batch (0-based, reassigned on compact).
    pub slot_id: usize,
    /// The request occupying this slot, if any.
    pub request: Option<GenerationRequest>,
    /// Number of tokens generated so far.
    pub current_position: usize,
    /// Maximum tokens this slot will generate (copied from request).
    pub max_tokens: usize,
    /// Whether this slot is actively generating.
    pub is_active: bool,
    /// Tokens generated in this slot (output buffer).
    generated_tokens: Vec<u32>,
}

impl GenerationSlot {
    fn empty(slot_id: usize) -> Self {
        Self {
            slot_id,
            request: None,
            current_position: 0,
            max_tokens: 0,
            is_active: false,
            generated_tokens: Vec::new(),
        }
    }

    /// Whether the slot has reached its generation limit.
    pub fn is_finished(&self) -> bool {
        self.is_active && self.current_position >= self.max_tokens
    }

    /// Read-only access to generated tokens.
    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated_tokens
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the batching engine.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchError {
    /// All batch slots are occupied.
    BatchFull,
    /// The specified slot was not found or is not active.
    SlotNotFound(usize),
    /// The request ID was not found.
    RequestNotFound(u64),
    /// Preemption is disabled or no suitable candidate exists.
    PreemptionFailed,
    /// Configuration / parameter error.
    ConfigError(String),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchFull => write!(f, "all batch slots are occupied"),
            Self::SlotNotFound(id) => {
                write!(f, "slot {id} not found or inactive")
            }
            Self::RequestNotFound(id) => write!(f, "request {id} not found"),
            Self::PreemptionFailed => {
                write!(f, "preemption failed or disabled")
            }
            Self::ConfigError(msg) => write!(f, "config error: {msg}"),
        }
    }
}

impl std::error::Error for BatchError {}

// ---------------------------------------------------------------------------
// SlotManager
// ---------------------------------------------------------------------------

/// Manages active and free generation slots.
#[derive(Debug)]
pub struct SlotManager {
    slots: Vec<GenerationSlot>,
    config: ContinuousBatchConfig,
    next_request_id: u64,
    stats: ContinuousBatchStats,
}

impl SlotManager {
    /// Create a new slot manager.
    pub fn new(config: ContinuousBatchConfig) -> Self {
        let slots = (0..config.max_batch_size).map(GenerationSlot::empty).collect();
        Self { slots, config, next_request_id: 1, stats: ContinuousBatchStats::default() }
    }

    /// Insert a new request into the first available slot.
    ///
    /// Returns `(slot_id, request_id)`.
    pub fn insert(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        priority: u32,
    ) -> Result<(usize, u64), BatchError> {
        if prompt_tokens.is_empty() {
            return Err(BatchError::ConfigError("prompt must not be empty".into()));
        }
        if max_tokens == 0 {
            return Err(BatchError::ConfigError("max_tokens must be > 0".into()));
        }

        let slot = self.slots.iter_mut().find(|s| !s.is_active).ok_or(BatchError::BatchFull)?;

        let request_id = self.next_request_id;
        self.next_request_id += 1;

        slot.request = Some(GenerationRequest { request_id, prompt_tokens, max_tokens, priority });
        slot.current_position = 0;
        slot.max_tokens = max_tokens;
        slot.is_active = true;
        slot.generated_tokens.clear();

        let slot_id = slot.slot_id;
        Ok((slot_id, request_id))
    }

    /// Remove a request by slot index, freeing the slot.
    pub fn remove(&mut self, slot_id: usize) -> Result<GenerationRequest, BatchError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.slot_id == slot_id && s.is_active)
            .ok_or(BatchError::SlotNotFound(slot_id))?;

        let request = slot.request.take().ok_or(BatchError::SlotNotFound(slot_id))?;
        let gen_len = slot.generated_tokens.len() as u64;

        slot.is_active = false;
        slot.current_position = 0;
        slot.max_tokens = 0;
        slot.generated_tokens.clear();

        self.stats.total_completed += 1;
        self.stats.total_tokens_generated += gen_len;
        let n = self.stats.total_completed as f64;
        self.stats.avg_generation_length =
            self.stats.avg_generation_length * ((n - 1.0) / n) + gen_len as f64 / n;

        Ok(request)
    }

    /// Remove a request by its request ID.
    pub fn remove_by_request_id(
        &mut self,
        request_id: u64,
    ) -> Result<GenerationRequest, BatchError> {
        let slot_id = self
            .slots
            .iter()
            .find(|s| s.is_active && s.request.as_ref().is_some_and(|r| r.request_id == request_id))
            .map(|s| s.slot_id)
            .ok_or(BatchError::RequestNotFound(request_id))?;
        self.remove(slot_id)
    }

    /// Compact: move all active slots to the front, reassign IDs.
    pub fn compact(&mut self) {
        self.slots.sort_by(|a, b| b.is_active.cmp(&a.is_active));
        for (i, slot) in self.slots.iter_mut().enumerate() {
            slot.slot_id = i;
        }
    }

    /// Number of active slots.
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_active).count()
    }

    /// Whether all slots are occupied.
    pub fn is_full(&self) -> bool {
        self.active_count() == self.config.max_batch_size
    }

    /// Get a reference to a slot by its current slot_id.
    pub fn get_slot(&self, slot_id: usize) -> Option<&GenerationSlot> {
        self.slots.iter().find(|s| s.slot_id == slot_id)
    }

    /// Get a mutable reference to a slot by its current slot_id.
    pub fn get_slot_mut(&mut self, slot_id: usize) -> Option<&mut GenerationSlot> {
        self.slots.iter_mut().find(|s| s.slot_id == slot_id)
    }

    /// All slots (active and inactive).
    pub fn slots(&self) -> &[GenerationSlot] {
        &self.slots
    }

    /// Engine configuration.
    pub fn config(&self) -> &ContinuousBatchConfig {
        &self.config
    }

    /// Accumulated stats (call [`Self::compute_stats`] for a utilisation snapshot).
    pub fn stats(&self) -> &ContinuousBatchStats {
        &self.stats
    }

    /// Snapshot of stats with live utilisation.
    pub fn compute_stats(&self) -> ContinuousBatchStats {
        let utilization = if self.config.max_batch_size > 0 {
            self.active_count() as f64 / self.config.max_batch_size as f64
        } else {
            0.0
        };
        ContinuousBatchStats {
            slot_utilization: utilization,
            preemption_count: self.stats.preemption_count,
            avg_generation_length: self.stats.avg_generation_length,
            total_completed: self.stats.total_completed,
            total_tokens_generated: self.stats.total_tokens_generated,
        }
    }

    fn record_preemption(&mut self) {
        self.stats.preemption_count += 1;
    }
}

// ---------------------------------------------------------------------------
// IterationBatch
// ---------------------------------------------------------------------------

/// Tokens for the current iteration across all active slots.
#[derive(Debug, Clone, Default)]
pub struct IterationBatch {
    /// Slot indices included in this iteration.
    pub slot_indices: Vec<usize>,
    /// Token ID for each included slot.
    pub tokens: Vec<u32>,
    /// Sequence position for each slot.
    pub positions: Vec<usize>,
}

impl IterationBatch {
    /// Number of slots in this iteration.
    pub fn len(&self) -> usize {
        self.slot_indices.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.slot_indices.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SlotScheduler
// ---------------------------------------------------------------------------

/// Decides which slots to include in each iteration.
pub struct SlotScheduler;

impl SlotScheduler {
    /// Build an [`IterationBatch`] from the current slot manager state.
    ///
    /// Includes every active, non-finished slot.  Each slot contributes its
    /// most recent generated token, or the first prompt token if none have
    /// been generated yet.
    pub fn build_iteration(manager: &SlotManager) -> IterationBatch {
        let mut batch = IterationBatch::default();
        for slot in manager.slots() {
            if !slot.is_active || slot.is_finished() {
                continue;
            }
            let token = if let Some(&last) = slot.generated_tokens.last() {
                last
            } else if let Some(req) = &slot.request {
                req.prompt_tokens.first().copied().unwrap_or(0)
            } else {
                0
            };
            batch.slot_indices.push(slot.slot_id);
            batch.tokens.push(token);
            batch.positions.push(slot.current_position);
        }
        batch
    }
}

// ---------------------------------------------------------------------------
// PreemptionPolicy
// ---------------------------------------------------------------------------

/// Policy for preempting low-priority slots when memory is scarce.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreemptionPolicy {
    /// Never preempt.
    Disabled,
    /// Preempt the lowest-priority active slot.
    LowestPriority,
    /// Preempt the slot with the shortest generation so far.
    ShortestGeneration,
}

impl PreemptionPolicy {
    /// Find a slot to preempt whose priority is strictly below
    /// `new_priority`.  Returns the victim's `slot_id`.
    pub fn select_victim(&self, manager: &SlotManager, new_priority: u32) -> Option<usize> {
        match self {
            Self::Disabled => None,
            Self::LowestPriority => manager
                .slots()
                .iter()
                .filter(|s| {
                    s.is_active && s.request.as_ref().is_some_and(|r| r.priority < new_priority)
                })
                .min_by_key(|s| s.request.as_ref().map(|r| r.priority).unwrap_or(u32::MAX))
                .map(|s| s.slot_id),
            Self::ShortestGeneration => manager
                .slots()
                .iter()
                .filter(|s| {
                    s.is_active && s.request.as_ref().is_some_and(|r| r.priority < new_priority)
                })
                .min_by_key(|s| s.current_position)
                .map(|s| s.slot_id),
        }
    }

    /// Preempt a victim and insert a new request in its place.
    ///
    /// Returns `(new_slot_id, new_request_id, evicted_request)`.
    pub fn preempt_and_insert(
        &self,
        manager: &mut SlotManager,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        priority: u32,
    ) -> Result<(usize, u64, GenerationRequest), BatchError> {
        let victim = self.select_victim(manager, priority).ok_or(BatchError::PreemptionFailed)?;
        let evicted = manager.remove(victim)?;
        manager.record_preemption();
        let (slot_id, req_id) = manager.insert(prompt_tokens, max_tokens, priority)?;
        Ok((slot_id, req_id, evicted))
    }
}

// ---------------------------------------------------------------------------
// CompactBatcher
// ---------------------------------------------------------------------------

/// Removes gaps from finished slots and maintains contiguous batch layout.
pub struct CompactBatcher;

impl CompactBatcher {
    /// Remove all finished slots and compact the batch.
    ///
    /// Returns the completed requests.
    pub fn compact_finished(manager: &mut SlotManager) -> Vec<GenerationRequest> {
        let finished: Vec<usize> =
            manager.slots().iter().filter(|s| s.is_finished()).map(|s| s.slot_id).collect();

        let mut completed = Vec::new();
        for id in finished {
            if let Ok(req) = manager.remove(id) {
                completed.push(req);
            }
        }
        manager.compact();
        completed
    }

    /// Compact without removing finished slots.
    pub fn compact_gaps(manager: &mut SlotManager) {
        manager.compact();
    }

    /// Check whether active slots are contiguous (no gaps).
    pub fn is_contiguous(manager: &SlotManager) -> bool {
        let active = manager.active_count();
        if active == 0 {
            return true;
        }
        manager.slots().iter().take(active).all(|s| s.is_active)
    }
}

// ---------------------------------------------------------------------------
// ContinuousBatchStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the continuous batching engine.
#[derive(Debug, Clone, Default)]
pub struct ContinuousBatchStats {
    /// Current slot utilisation (0.0–1.0).
    pub slot_utilization: f64,
    /// Number of preemptions performed.
    pub preemption_count: u64,
    /// Average generation length across completed requests.
    pub avg_generation_length: f64,
    /// Total requests completed.
    pub total_completed: u64,
    /// Total tokens generated across all completed requests.
    pub total_tokens_generated: u64,
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU reference: advance one generation step for the given slot.
pub fn cpu_step_slot(
    manager: &mut SlotManager,
    slot_id: usize,
    logits: &[f32],
) -> Result<u32, BatchError> {
    let max_seq_len = manager.config().max_seq_len;
    let slot = manager
        .get_slot_mut(slot_id)
        .filter(|s| s.is_active && !s.is_finished())
        .ok_or(BatchError::SlotNotFound(slot_id))?;

    let token = cpu_argmax(logits);

    let prompt_len = slot.request.as_ref().map(|r| r.prompt_tokens.len()).unwrap_or(0);
    if prompt_len + slot.current_position + 1 > max_seq_len {
        slot.current_position = slot.max_tokens; // mark finished
        return Ok(token);
    }

    slot.generated_tokens.push(token);
    slot.current_position += 1;
    Ok(token)
}

/// CPU reference: advance one iteration for every active, non-finished slot.
pub fn cpu_step_iteration(
    manager: &mut SlotManager,
    logits_per_slot: &[Vec<f32>],
) -> Result<Vec<(usize, u32)>, BatchError> {
    let active_ids: Vec<usize> = manager
        .slots()
        .iter()
        .filter(|s| s.is_active && !s.is_finished())
        .map(|s| s.slot_id)
        .collect();

    if active_ids.len() > logits_per_slot.len() {
        return Err(BatchError::ConfigError("not enough logits for active slots".into()));
    }

    let mut results = Vec::new();
    for (i, &sid) in active_ids.iter().enumerate() {
        let tok = cpu_step_slot(manager, sid, &logits_per_slot[i])?;
        results.push((sid, tok));
    }
    Ok(results)
}

/// CPU reference: argmax over a logit slice.  Returns 0 for empty slices.
pub fn cpu_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Format a human-readable status line.
pub fn format_status(manager: &SlotManager) -> String {
    let stats = manager.compute_stats();
    format!(
        "ContinuousBatch(active={}/{}, util={:.1}%, \
         completed={}, tokens={}, preemptions={})",
        manager.active_count(),
        manager.config().max_batch_size,
        stats.slot_utilization * 100.0,
        stats.total_completed,
        stats.total_tokens_generated,
        stats.preemption_count,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg4() -> ContinuousBatchConfig {
        ContinuousBatchConfig { max_batch_size: 4, max_seq_len: 128, iteration_timeout_ms: 100 }
    }

    fn logits(best: usize, len: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        if best < len {
            v[best] = 1.0;
        }
        v
    }

    // == SlotManager creation ===============================================

    #[test]
    fn test_create_manager_empty_slots() {
        let mgr = SlotManager::new(cfg4());
        assert_eq!(mgr.slots().len(), 4);
        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.slots().iter().all(|s| !s.is_active));
    }

    #[test]
    fn test_create_manager_slot_ids() {
        let mgr = SlotManager::new(cfg4());
        for (i, s) in mgr.slots().iter().enumerate() {
            assert_eq!(s.slot_id, i);
        }
    }

    // == Insert =============================================================

    #[test]
    fn test_insert_returns_slot_and_id() {
        let mut mgr = SlotManager::new(cfg4());
        let (slot, rid) = mgr.insert(vec![1, 2, 3], 10, 5).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(rid, 1);
    }

    #[test]
    fn test_insert_fills_first_free_slot() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        let (slot, _) = mgr.insert(vec![2], 5, 0).unwrap();
        assert_eq!(slot, 1);
    }

    #[test]
    fn test_insert_multiple_requests() {
        let mut mgr = SlotManager::new(cfg4());
        for _ in 0..4 {
            mgr.insert(vec![1], 5, 0).unwrap();
        }
        assert_eq!(mgr.active_count(), 4);
    }

    #[test]
    fn test_insert_returns_unique_ids() {
        let mut mgr = SlotManager::new(cfg4());
        let ids: Vec<u64> = (0..4).map(|_| mgr.insert(vec![1], 5, 0).unwrap().1).collect();
        let set: std::collections::HashSet<u64> = ids.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_insert_empty_prompt_error() {
        let mut mgr = SlotManager::new(cfg4());
        let err = mgr.insert(vec![], 5, 0).unwrap_err();
        assert!(matches!(err, BatchError::ConfigError(_)));
    }

    #[test]
    fn test_insert_zero_max_tokens_error() {
        let mut mgr = SlotManager::new(cfg4());
        let err = mgr.insert(vec![1], 0, 0).unwrap_err();
        assert!(matches!(err, BatchError::ConfigError(_)));
    }

    #[test]
    fn test_insert_batch_full() {
        let mut mgr = SlotManager::new(cfg4());
        for _ in 0..4 {
            mgr.insert(vec![1], 5, 0).unwrap();
        }
        assert_eq!(mgr.insert(vec![1], 5, 0).unwrap_err(), BatchError::BatchFull);
    }

    #[test]
    fn test_is_full() {
        let mut mgr = SlotManager::new(cfg4());
        assert!(!mgr.is_full());
        for _ in 0..4 {
            mgr.insert(vec![1], 5, 0).unwrap();
        }
        assert!(mgr.is_full());
    }

    // == Remove =============================================================

    #[test]
    fn test_remove_by_slot_id() {
        let mut mgr = SlotManager::new(cfg4());
        let (sid, _) = mgr.insert(vec![1, 2], 5, 0).unwrap();
        let req = mgr.remove(sid).unwrap();
        assert_eq!(req.prompt_tokens, vec![1, 2]);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_remove_frees_slot_for_reuse() {
        let mut mgr = SlotManager::new(cfg4());
        let (sid, _) = mgr.insert(vec![1], 5, 0).unwrap();
        mgr.remove(sid).unwrap();
        let (sid2, _) = mgr.insert(vec![2], 5, 0).unwrap();
        assert_eq!(sid2, sid);
    }

    #[test]
    fn test_remove_nonexistent_slot() {
        let mut mgr = SlotManager::new(cfg4());
        assert_eq!(mgr.remove(99).unwrap_err(), BatchError::SlotNotFound(99));
    }

    #[test]
    fn test_remove_inactive_slot() {
        let mut mgr = SlotManager::new(cfg4());
        assert_eq!(mgr.remove(0).unwrap_err(), BatchError::SlotNotFound(0));
    }

    #[test]
    fn test_remove_by_request_id() {
        let mut mgr = SlotManager::new(cfg4());
        let (_, rid) = mgr.insert(vec![10, 20], 5, 3).unwrap();
        let req = mgr.remove_by_request_id(rid).unwrap();
        assert_eq!(req.request_id, rid);
        assert_eq!(req.priority, 3);
    }

    #[test]
    fn test_remove_by_request_id_not_found() {
        let mut mgr = SlotManager::new(cfg4());
        assert_eq!(mgr.remove_by_request_id(42).unwrap_err(), BatchError::RequestNotFound(42));
    }

    // == Insert into freed slot (gap filling) ===============================

    #[test]
    fn test_new_request_fills_gap() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![1], 5, 0).unwrap();
        let (_s1, _) = mgr.insert(vec![2], 5, 0).unwrap();
        mgr.remove(s0).unwrap();
        let (s_new, _) = mgr.insert(vec![3], 5, 0).unwrap();
        assert_eq!(s_new, s0); // reuses freed slot
    }

    #[test]
    fn test_insert_after_removal_reuses_slot() {
        let mut mgr = SlotManager::new(ContinuousBatchConfig {
            max_batch_size: 1,
            max_seq_len: 64,
            iteration_timeout_ms: 50,
        });
        let (sid, _) = mgr.insert(vec![1], 5, 0).unwrap();
        mgr.remove(sid).unwrap();
        let (sid2, _) = mgr.insert(vec![2], 5, 0).unwrap();
        assert_eq!(sid2, sid);
    }

    // == Finished detection =================================================

    #[test]
    fn test_finished_slot_detection() {
        let mut mgr = SlotManager::new(cfg4());
        let (sid, _) = mgr.insert(vec![1], 2, 0).unwrap();
        cpu_step_slot(&mut mgr, sid, &logits(3, 8)).unwrap();
        assert!(!mgr.get_slot(sid).unwrap().is_finished());
        cpu_step_slot(&mut mgr, sid, &logits(4, 8)).unwrap();
        assert!(mgr.get_slot(sid).unwrap().is_finished());
    }

    #[test]
    fn test_compact_finished_removes_done() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![1], 1, 0).unwrap();
        cpu_step_slot(&mut mgr, s0, &logits(0, 8)).unwrap();
        assert!(mgr.get_slot(s0).unwrap().is_finished());

        let completed = CompactBatcher::compact_finished(&mut mgr);
        assert_eq!(completed.len(), 1);
        assert_eq!(mgr.active_count(), 0);
    }

    // == Multiple requests at different positions ===========================

    #[test]
    fn test_multiple_requests_different_positions() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![1], 10, 0).unwrap();
        let (s1, _) = mgr.insert(vec![2], 10, 0).unwrap();
        // Advance s0 by 3 steps, s1 by 1 step
        for _ in 0..3 {
            cpu_step_slot(&mut mgr, s0, &logits(1, 8)).unwrap();
        }
        cpu_step_slot(&mut mgr, s1, &logits(2, 8)).unwrap();
        assert_eq!(mgr.get_slot(s0).unwrap().current_position, 3);
        assert_eq!(mgr.get_slot(s1).unwrap().current_position, 1);
    }

    #[test]
    fn test_step_advances_position() {
        let mut mgr = SlotManager::new(cfg4());
        let (sid, _) = mgr.insert(vec![1], 10, 0).unwrap();
        cpu_step_slot(&mut mgr, sid, &logits(5, 8)).unwrap();
        assert_eq!(mgr.get_slot(sid).unwrap().current_position, 1);
        assert_eq!(mgr.get_slot(sid).unwrap().generated_tokens(), &[5]);
    }

    #[test]
    fn test_step_multiple_slots() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![1], 10, 0).unwrap();
        let (s1, _) = mgr.insert(vec![2], 10, 0).unwrap();
        let t0 = cpu_step_slot(&mut mgr, s0, &logits(3, 8)).unwrap();
        let t1 = cpu_step_slot(&mut mgr, s1, &logits(7, 8)).unwrap();
        assert_eq!(t0, 3);
        assert_eq!(t1, 7);
    }

    // == Preemption =========================================================

    #[test]
    fn test_preemption_disabled() {
        let mgr = SlotManager::new(cfg4());
        assert_eq!(PreemptionPolicy::Disabled.select_victim(&mgr, 100), None);
    }

    #[test]
    fn test_preemption_lowest_priority() {
        let mut mgr = SlotManager::new(cfg4());
        let (s_low, _) = mgr.insert(vec![1], 5, 1).unwrap();
        mgr.insert(vec![2], 5, 5).unwrap();
        mgr.insert(vec![3], 5, 10).unwrap();
        let victim = PreemptionPolicy::LowestPriority.select_victim(&mgr, 8);
        assert_eq!(victim, Some(s_low));
    }

    #[test]
    fn test_preemption_shortest_generation() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![1], 10, 1).unwrap();
        let (s1, _) = mgr.insert(vec![2], 10, 2).unwrap();
        // Advance s0 by 5 steps so s1 is shorter
        for _ in 0..5 {
            cpu_step_slot(&mut mgr, s0, &logits(0, 8)).unwrap();
        }
        cpu_step_slot(&mut mgr, s1, &logits(0, 8)).unwrap();
        let victim = PreemptionPolicy::ShortestGeneration.select_victim(&mgr, 100);
        assert_eq!(victim, Some(s1));
    }

    #[test]
    fn test_preemption_no_victim_all_higher() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 10).unwrap();
        assert_eq!(PreemptionPolicy::LowestPriority.select_victim(&mgr, 5), None);
    }

    #[test]
    fn test_preempt_and_insert() {
        let mut mgr = SlotManager::new(ContinuousBatchConfig {
            max_batch_size: 1,
            max_seq_len: 128,
            iteration_timeout_ms: 100,
        });
        mgr.insert(vec![1], 5, 1).unwrap();
        assert!(mgr.is_full());

        let (sid, rid, evicted) =
            PreemptionPolicy::LowestPriority.preempt_and_insert(&mut mgr, vec![2], 5, 10).unwrap();
        assert_eq!(sid, 0);
        assert!(rid > 0);
        assert_eq!(evicted.priority, 1);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_preemption_count_tracked() {
        let mut mgr = SlotManager::new(ContinuousBatchConfig {
            max_batch_size: 1,
            max_seq_len: 128,
            iteration_timeout_ms: 100,
        });
        mgr.insert(vec![1], 5, 1).unwrap();
        PreemptionPolicy::LowestPriority.preempt_and_insert(&mut mgr, vec![2], 5, 10).unwrap();
        assert_eq!(mgr.stats().preemption_count, 1);
    }

    #[test]
    fn test_preempt_and_insert_fails_disabled() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 1).unwrap();
        let err =
            PreemptionPolicy::Disabled.preempt_and_insert(&mut mgr, vec![2], 5, 10).unwrap_err();
        assert_eq!(err, BatchError::PreemptionFailed);
    }

    // == Compaction ==========================================================

    #[test]
    fn test_compact_no_gaps() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        mgr.insert(vec![2], 5, 0).unwrap();
        CompactBatcher::compact_gaps(&mut mgr);
        assert!(CompactBatcher::is_contiguous(&mgr));
    }

    #[test]
    fn test_compact_fills_gap() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        let (s1, _) = mgr.insert(vec![2], 5, 0).unwrap();
        mgr.insert(vec![3], 5, 0).unwrap();
        mgr.remove(s1).unwrap(); // gap at index 1
        assert!(!CompactBatcher::is_contiguous(&mgr));
        CompactBatcher::compact_gaps(&mut mgr);
        assert!(CompactBatcher::is_contiguous(&mgr));
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_compact_multiple_gaps() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![1], 5, 0).unwrap();
        mgr.insert(vec![2], 5, 0).unwrap();
        let (s2, _) = mgr.insert(vec![3], 5, 0).unwrap();
        mgr.insert(vec![4], 5, 0).unwrap();
        mgr.remove(s0).unwrap();
        mgr.remove(s2).unwrap();
        CompactBatcher::compact_gaps(&mut mgr);
        assert!(CompactBatcher::is_contiguous(&mgr));
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_compact_preserves_active_data() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![10], 5, 0).unwrap();
        mgr.insert(vec![20], 5, 0).unwrap();
        cpu_step_slot(&mut mgr, s0, &logits(7, 8)).unwrap();
        mgr.remove(s0).unwrap();
        CompactBatcher::compact_gaps(&mut mgr);
        // The remaining slot should still have its prompt
        let active: Vec<&GenerationSlot> = mgr.slots().iter().filter(|s| s.is_active).collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].request.as_ref().unwrap().prompt_tokens, vec![20]);
    }

    // == Contiguous layout ==================================================

    #[test]
    fn test_contiguous_after_compact() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        let (s1, _) = mgr.insert(vec![2], 5, 0).unwrap();
        mgr.insert(vec![3], 5, 0).unwrap();
        mgr.remove(s1).unwrap();
        CompactBatcher::compact_gaps(&mut mgr);
        assert!(CompactBatcher::is_contiguous(&mgr));
    }

    #[test]
    fn test_not_contiguous_with_gap() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        let (s1, _) = mgr.insert(vec![2], 5, 0).unwrap();
        mgr.insert(vec![3], 5, 0).unwrap();
        mgr.remove(s1).unwrap();
        assert!(!CompactBatcher::is_contiguous(&mgr));
    }

    #[test]
    fn test_contiguous_empty_batch() {
        let mgr = SlotManager::new(cfg4());
        assert!(CompactBatcher::is_contiguous(&mgr));
    }

    // == Stats ==============================================================

    #[test]
    fn test_stats_initial_zeros() {
        let mgr = SlotManager::new(cfg4());
        let s = mgr.compute_stats();
        assert_eq!(s.total_completed, 0);
        assert_eq!(s.total_tokens_generated, 0);
        assert_eq!(s.preemption_count, 0);
        assert!((s.slot_utilization).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_after_completion() {
        let mut mgr = SlotManager::new(cfg4());
        let (sid, _) = mgr.insert(vec![1], 2, 0).unwrap();
        cpu_step_slot(&mut mgr, sid, &logits(0, 8)).unwrap();
        cpu_step_slot(&mut mgr, sid, &logits(0, 8)).unwrap();
        CompactBatcher::compact_finished(&mut mgr);
        assert_eq!(mgr.stats().total_completed, 1);
        assert_eq!(mgr.stats().total_tokens_generated, 2);
    }

    #[test]
    fn test_stats_utilization() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        mgr.insert(vec![2], 5, 0).unwrap();
        let s = mgr.compute_stats();
        assert!((s.slot_utilization - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_avg_generation_length() {
        let mut mgr = SlotManager::new(cfg4());
        // Request 1: generate 2 tokens
        let (s0, _) = mgr.insert(vec![1], 2, 0).unwrap();
        cpu_step_slot(&mut mgr, s0, &logits(0, 8)).unwrap();
        cpu_step_slot(&mut mgr, s0, &logits(0, 8)).unwrap();
        CompactBatcher::compact_finished(&mut mgr);
        assert!((mgr.stats().avg_generation_length - 2.0).abs() < f64::EPSILON);

        // Request 2: generate 4 tokens
        let (s1, _) = mgr.insert(vec![2], 4, 0).unwrap();
        for _ in 0..4 {
            cpu_step_slot(&mut mgr, s1, &logits(0, 8)).unwrap();
        }
        CompactBatcher::compact_finished(&mut mgr);
        // avg = (2 + 4) / 2 = 3
        assert!((mgr.stats().avg_generation_length - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_multiple_completions() {
        let mut mgr = SlotManager::new(cfg4());
        for _ in 0..3 {
            let (sid, _) = mgr.insert(vec![1], 1, 0).unwrap();
            cpu_step_slot(&mut mgr, sid, &logits(0, 8)).unwrap();
            CompactBatcher::compact_finished(&mut mgr);
        }
        assert_eq!(mgr.stats().total_completed, 3);
        assert_eq!(mgr.stats().total_tokens_generated, 3);
    }

    #[test]
    fn test_stats_preemption_count() {
        let mut mgr = SlotManager::new(ContinuousBatchConfig {
            max_batch_size: 1,
            max_seq_len: 128,
            iteration_timeout_ms: 100,
        });
        mgr.insert(vec![1], 5, 1).unwrap();
        PreemptionPolicy::LowestPriority.preempt_and_insert(&mut mgr, vec![2], 5, 10).unwrap();
        PreemptionPolicy::LowestPriority.preempt_and_insert(&mut mgr, vec![3], 5, 20).unwrap();
        assert_eq!(mgr.stats().preemption_count, 2);
    }

    // == Edge cases ==========================================================

    #[test]
    fn test_single_slot_batch() {
        let mut mgr = SlotManager::new(ContinuousBatchConfig {
            max_batch_size: 1,
            max_seq_len: 64,
            iteration_timeout_ms: 50,
        });
        let (sid, _) = mgr.insert(vec![1], 3, 0).unwrap();
        assert!(mgr.is_full());
        mgr.remove(sid).unwrap();
        assert!(!mgr.is_full());
        mgr.insert(vec![2], 3, 0).unwrap();
    }

    #[test]
    fn test_single_slot_full_then_free() {
        let mut mgr = SlotManager::new(ContinuousBatchConfig {
            max_batch_size: 1,
            max_seq_len: 64,
            iteration_timeout_ms: 50,
        });
        let (sid, _) = mgr.insert(vec![1], 1, 0).unwrap();
        cpu_step_slot(&mut mgr, sid, &logits(0, 8)).unwrap();
        CompactBatcher::compact_finished(&mut mgr);
        assert!(!mgr.is_full());
        mgr.insert(vec![2], 1, 0).unwrap();
    }

    #[test]
    fn test_all_slots_full() {
        let mut mgr = SlotManager::new(cfg4());
        for i in 0..4 {
            mgr.insert(vec![i as u32 + 1], 5, 0).unwrap();
        }
        assert!(mgr.is_full());
        assert_eq!(mgr.insert(vec![99], 5, 0).unwrap_err(), BatchError::BatchFull);
    }

    #[test]
    fn test_all_finish_simultaneously() {
        let mut mgr = SlotManager::new(cfg4());
        let mut sids = Vec::new();
        for _ in 0..4 {
            let (sid, _) = mgr.insert(vec![1], 1, 0).unwrap();
            sids.push(sid);
        }
        for sid in &sids {
            cpu_step_slot(&mut mgr, *sid, &logits(0, 8)).unwrap();
        }
        let completed = CompactBatcher::compact_finished(&mut mgr);
        assert_eq!(completed.len(), 4);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_max_seq_len_boundary() {
        let mut mgr = SlotManager::new(ContinuousBatchConfig {
            max_batch_size: 4,
            max_seq_len: 3, // prompt(1) + 2 generated = 3
            iteration_timeout_ms: 100,
        });
        let (sid, _) = mgr.insert(vec![1], 10, 0).unwrap();
        cpu_step_slot(&mut mgr, sid, &logits(0, 8)).unwrap(); // total=2
        cpu_step_slot(&mut mgr, sid, &logits(0, 8)).unwrap(); // total=3
        // Next step exceeds max_seq_len
        cpu_step_slot(&mut mgr, sid, &logits(0, 8)).unwrap();
        assert!(mgr.get_slot(sid).unwrap().is_finished());
    }

    #[test]
    fn test_large_batch_size() {
        let mut mgr = SlotManager::new(ContinuousBatchConfig {
            max_batch_size: 64,
            max_seq_len: 1024,
            iteration_timeout_ms: 200,
        });
        for _ in 0..64 {
            mgr.insert(vec![1], 5, 0).unwrap();
        }
        assert!(mgr.is_full());
        assert_eq!(mgr.active_count(), 64);
    }

    // == Property tests =====================================================

    #[test]
    fn test_property_active_count_bounded() {
        let mut mgr = SlotManager::new(cfg4());
        for _ in 0..4 {
            mgr.insert(vec![1], 5, 0).unwrap();
        }
        assert!(mgr.active_count() <= 4);
        assert!(mgr.insert(vec![1], 5, 0).is_err());
    }

    #[test]
    fn test_property_insert_remove_balanced() {
        let mut mgr = SlotManager::new(cfg4());
        let mut ids = Vec::new();
        for _ in 0..4 {
            ids.push(mgr.insert(vec![1], 5, 0).unwrap().0);
        }
        for sid in ids {
            mgr.remove(sid).unwrap();
        }
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_property_tokens_monotonic() {
        let mut mgr = SlotManager::new(cfg4());
        let (sid, _) = mgr.insert(vec![1], 20, 0).unwrap();
        let mut prev = 0;
        for i in 0..10 {
            cpu_step_slot(&mut mgr, sid, &logits(i % 8, 8)).unwrap();
            let pos = mgr.get_slot(sid).unwrap().current_position;
            assert!(pos > prev);
            prev = pos;
        }
    }

    // == SlotScheduler ======================================================

    #[test]
    fn test_scheduler_builds_iteration_batch() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        mgr.insert(vec![2], 5, 0).unwrap();
        let batch = SlotScheduler::build_iteration(&mgr);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_scheduler_excludes_finished_slots() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![1], 1, 0).unwrap();
        mgr.insert(vec![2], 5, 0).unwrap();
        cpu_step_slot(&mut mgr, s0, &logits(0, 8)).unwrap();
        assert!(mgr.get_slot(s0).unwrap().is_finished());
        let batch = SlotScheduler::build_iteration(&mgr);
        assert_eq!(batch.len(), 1);
        assert!(!batch.slot_indices.contains(&s0));
    }

    #[test]
    fn test_scheduler_empty_batch() {
        let mgr = SlotManager::new(cfg4());
        let batch = SlotScheduler::build_iteration(&mgr);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_scheduler_tokens_from_prompt() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![42], 5, 0).unwrap();
        let batch = SlotScheduler::build_iteration(&mgr);
        assert_eq!(batch.tokens[0], 42);
    }

    #[test]
    fn test_scheduler_tokens_from_generated() {
        let mut mgr = SlotManager::new(cfg4());
        let (sid, _) = mgr.insert(vec![1], 5, 0).unwrap();
        cpu_step_slot(&mut mgr, sid, &logits(7, 8)).unwrap();
        let batch = SlotScheduler::build_iteration(&mgr);
        assert_eq!(batch.tokens[0], 7);
    }

    // == CPU reference ======================================================

    #[test]
    fn test_cpu_step_iteration() {
        let mut mgr = SlotManager::new(cfg4());
        let (s0, _) = mgr.insert(vec![1], 5, 0).unwrap();
        let (s1, _) = mgr.insert(vec![2], 5, 0).unwrap();
        let logits_list = vec![logits(3, 8), logits(6, 8)];
        let results = cpu_step_iteration(&mut mgr, &logits_list).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (s0, 3));
        assert_eq!(results[1], (s1, 6));
    }

    #[test]
    fn test_cpu_step_iteration_not_enough_logits() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        mgr.insert(vec![2], 5, 0).unwrap();
        let err = cpu_step_iteration(&mut mgr, &[logits(0, 8)]).unwrap_err();
        assert!(matches!(err, BatchError::ConfigError(_)));
    }

    // == Argmax =============================================================

    #[test]
    fn test_argmax_basic() {
        assert_eq!(cpu_argmax(&[0.1, 0.9, 0.2]), 1);
    }

    #[test]
    fn test_argmax_empty() {
        assert_eq!(cpu_argmax(&[]), 0);
    }

    #[test]
    fn test_argmax_negative() {
        assert_eq!(cpu_argmax(&[-1.0, -0.5, -2.0]), 1);
    }

    // == BatchError display =================================================

    #[test]
    fn test_batch_error_display() {
        assert_eq!(format!("{}", BatchError::BatchFull), "all batch slots are occupied");
        assert!(format!("{}", BatchError::SlotNotFound(7)).contains("7"));
        assert!(format!("{}", BatchError::RequestNotFound(42)).contains("42"));
        assert_eq!(format!("{}", BatchError::PreemptionFailed), "preemption failed or disabled");
        assert!(format!("{}", BatchError::ConfigError("bad".into())).contains("bad"));
    }

    // == Format status ======================================================

    #[test]
    fn test_format_status() {
        let mut mgr = SlotManager::new(cfg4());
        mgr.insert(vec![1], 5, 0).unwrap();
        let s = format_status(&mgr);
        assert!(s.contains("active=1/4"));
        assert!(s.contains("util=25.0%"));
    }

    #[test]
    fn test_format_status_empty() {
        let mgr = SlotManager::new(cfg4());
        let s = format_status(&mgr);
        assert!(s.contains("active=0/4"));
    }

    // == IterationBatch helpers =============================================

    #[test]
    fn test_iteration_batch_len_empty() {
        let b = IterationBatch::default();
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }
}
