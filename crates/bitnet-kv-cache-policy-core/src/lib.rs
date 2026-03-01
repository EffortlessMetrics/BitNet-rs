//! KV-cache update decision policy core.
//!
//! This microcrate keeps sequence-length transition logic isolated from any
//! tensor backend. Callers can map the returned action to concrete operations
//! (append, truncate, replace, or no-op) in their own cache representations.

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
pub fn decide_update_action(current_len: usize, seq_len: usize) -> KvCacheUpdateAction {
    match (current_len, seq_len) {
        (0, target) => KvCacheUpdateAction::Initialize { seq_len: target },
        (curr, next) if next > curr => {
            KvCacheUpdateAction::Append { new_tokens: next.saturating_sub(curr) }
        }
        (curr, next) if next < curr => KvCacheUpdateAction::Truncate { target_len: next },
        (_, len) => KvCacheUpdateAction::ReplaceSameLen { len },
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
