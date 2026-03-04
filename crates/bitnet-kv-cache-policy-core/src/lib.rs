//! KV-cache update and eviction policy core.
//!
//! This microcrate keeps cache transition and eviction logic isolated from any
//! tensor backend. Callers can map the returned action/picks to concrete
//! operations (append, truncate, replace, or no-op) in their own cache
//! representations.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Logical action implied by transitioning from one sequence length to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheUpdateAction {
    /// Sequence length increased; append exactly `new_tokens` worth of state.
    Append { new_tokens: usize },
    /// Sequence length decreased; truncate cache to `target_len`.
    Truncate { target_len: usize },
    /// Sequence length unchanged but data should be replaced for safety.
    ReplaceSameLen { len: usize },
    /// Cache was empty and should be initialized with `seq_len` tokens.
    Initialize { seq_len: usize },
}

/// Decide cache update action from sequence-length transition.
#[must_use]
pub const fn decide_update_action(current_len: usize, seq_len: usize) -> KvCacheUpdateAction {
    match (current_len, seq_len) {
        (0, target) => KvCacheUpdateAction::Initialize { seq_len: target },
        (curr, next) if next > curr => {
            KvCacheUpdateAction::Append { new_tokens: next.saturating_sub(curr) }
        }
        (curr, next) if next < curr => KvCacheUpdateAction::Truncate { target_len: next },
        (_, len) => KvCacheUpdateAction::ReplaceSameLen { len },
    }
}

/// Policy used to choose which KV entries to evict when a cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least-recently-used: evict the entry that was appended earliest.
    Lru,
    /// First-in-first-out (identical to LRU for append-only caches).
    Fifo,
    /// Evict the entry with the lowest cumulative attention score.
    AttentionScore,
    /// Hybrid: combine recency and attention score (configurable weight).
    Hybrid {
        /// Weight ∈ [0, 1] for the attention-score component.
        /// `0.0` = pure LRU, `1.0` = pure attention-score.
        attention_weight: u8,
    },
}

impl EvictionPolicy {
    /// Create a hybrid policy with the given attention-score weight in `[0.0, 1.0]`.
    /// The weight is clamped and stored as a percentage (0–100).
    #[must_use]
    pub fn hybrid(attention_weight: f32) -> Self {
        let pct = (attention_weight.clamp(0.0, 1.0) * 100.0).round() as u8;
        Self::Hybrid { attention_weight: pct }
    }

    /// Return the attention-score weight as `f32` in `[0.0, 1.0]`.
    #[must_use]
    pub fn attention_weight_f32(&self) -> f32 {
        match self {
            Self::Lru | Self::Fifo => 0.0,
            Self::AttentionScore => 1.0,
            Self::Hybrid { attention_weight } => *attention_weight as f32 / 100.0,
        }
    }
}

impl fmt::Display for EvictionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lru => write!(f, "LRU"),
            Self::Fifo => write!(f, "FIFO"),
            Self::AttentionScore => write!(f, "AttentionScore"),
            Self::Hybrid { attention_weight } => {
                write!(f, "Hybrid(attn={:.0}%)", *attention_weight as f32)
            }
        }
    }
}

/// Manages eviction candidates for append-only KV caches.
#[derive(Debug)]
pub struct KvEviction {
    policy: EvictionPolicy,
    entries: Vec<EvictionEntry>,
    next_order: u64,
}

#[derive(Debug, Clone)]
struct EvictionEntry {
    position: usize,
    insertion_order: u64,
    attention_score: f32,
}

impl KvEviction {
    /// Create a new eviction manager with the given policy.
    #[must_use]
    pub fn new(policy: EvictionPolicy) -> Self {
        Self { policy, entries: Vec::new(), next_order: 0 }
    }

    /// Register a new entry at the given position.
    pub fn insert(&mut self, position: usize) {
        self.entries.push(EvictionEntry {
            position,
            insertion_order: self.next_order,
            attention_score: 0.0,
        });
        self.next_order += 1;
    }

    /// Update attention scores for all tracked entries.
    /// `scores` maps each tracked index (in insertion order) to a score.
    pub fn update_scores(&mut self, scores: &[f32]) {
        let n = scores.len().min(self.entries.len());
        for (idx, score) in scores.iter().copied().take(n).enumerate() {
            self.entries[idx].attention_score += score;
        }
    }

    /// Number of tracked entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the tracker is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Select `count` entries to evict according to the policy.
    /// Returns the *positions* of the entries to remove.
    #[must_use]
    pub fn select_evictions(&self, count: usize) -> Vec<usize> {
        if count == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let count = count.min(self.entries.len());

        let mut scored: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, e)| {
                let priority = match self.policy {
                    EvictionPolicy::Lru | EvictionPolicy::Fifo => e.insertion_order as f64,
                    EvictionPolicy::AttentionScore => e.attention_score as f64,
                    EvictionPolicy::Hybrid { attention_weight } => {
                        let w = attention_weight as f64 / 100.0;
                        let max_order = self
                            .entries
                            .iter()
                            .map(|entry| entry.insertion_order)
                            .max()
                            .unwrap_or(1) as f64;
                        let recency = if max_order > 0.0 {
                            e.insertion_order as f64 / max_order
                        } else {
                            0.0
                        };
                        let attn = e.attention_score as f64;
                        (1.0 - w) * recency + w * attn
                    }
                };
                (idx, priority)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(count).map(|(idx, _)| self.entries[*idx].position).collect()
    }

    /// Remove entries at the given positions from the tracker.
    pub fn remove_positions(&mut self, positions: &[usize]) {
        self.entries.retain(|entry| !positions.contains(&entry.position));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_when_current_is_zero() {
        assert_eq!(decide_update_action(0, 8), KvCacheUpdateAction::Initialize { seq_len: 8 });
    }

    #[test]
    fn append_when_sequence_grows() {
        assert_eq!(decide_update_action(7, 11), KvCacheUpdateAction::Append { new_tokens: 4 });
    }

    #[test]
    fn truncate_when_sequence_shrinks() {
        assert_eq!(decide_update_action(9, 3), KvCacheUpdateAction::Truncate { target_len: 3 });
    }

    #[test]
    fn replace_when_lengths_match() {
        assert_eq!(decide_update_action(5, 5), KvCacheUpdateAction::ReplaceSameLen { len: 5 });
    }

    #[test]
    fn eviction_policy_display_and_clamp() {
        assert_eq!(format!("{}", EvictionPolicy::Lru), "LRU");
        assert_eq!(format!("{}", EvictionPolicy::Fifo), "FIFO");
        assert_eq!(format!("{}", EvictionPolicy::AttentionScore), "AttentionScore");

        let hybrid = EvictionPolicy::hybrid(0.7);
        assert_eq!(hybrid, EvictionPolicy::Hybrid { attention_weight: 70 });
        assert!((hybrid.attention_weight_f32() - 0.70).abs() < 1e-6);

        assert_eq!(EvictionPolicy::hybrid(1.5), EvictionPolicy::Hybrid { attention_weight: 100 });
        assert_eq!(EvictionPolicy::hybrid(-0.5), EvictionPolicy::Hybrid { attention_weight: 0 });
    }

    #[test]
    fn eviction_tracker_lru() {
        let mut ev = KvEviction::new(EvictionPolicy::Lru);
        ev.insert(10);
        ev.insert(11);
        ev.insert(12);

        assert_eq!(ev.select_evictions(2), vec![10, 11]);
        ev.remove_positions(&[11]);
        assert_eq!(ev.select_evictions(2), vec![10, 12]);
    }

    #[test]
    fn eviction_tracker_attention() {
        let mut ev = KvEviction::new(EvictionPolicy::AttentionScore);
        for i in 0..5 {
            ev.insert(i);
        }
        ev.update_scores(&[0.1, 0.1, 10.0, 0.1, 10.0]);
        let evicted = ev.select_evictions(2);
        assert_eq!(evicted.len(), 2);
        for pos in evicted {
            assert!(pos != 2 && pos != 4);
        }
    }

    proptest::proptest! {
        #[test]
        fn action_matches_transition(current in 0usize..2048, next in 0usize..2048) {
            let action = decide_update_action(current, next);
            match action {
                KvCacheUpdateAction::Initialize { seq_len } => {
                    proptest::prop_assert_eq!(current, 0);
                    proptest::prop_assert_eq!(seq_len, next);
                }
                KvCacheUpdateAction::Append { new_tokens } => {
                    proptest::prop_assert!(next > current);
                    proptest::prop_assert_eq!(new_tokens, next - current);
                }
                KvCacheUpdateAction::Truncate { target_len } => {
                    proptest::prop_assert!(next < current);
                    proptest::prop_assert_eq!(target_len, next);
                }
                KvCacheUpdateAction::ReplaceSameLen { len } => {
                    proptest::prop_assert_eq!(next, current);
                    proptest::prop_assert_eq!(len, next);
                }
            }
        }
    }
}
