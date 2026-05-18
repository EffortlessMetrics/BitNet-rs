//! Fixed-capacity ring buffer for token IDs.
//!
//! Efficient circular buffer for streaming token storage with O(1)
//! push/pop and sliding window semantics.
//!
//! The implementation is split into focused submodules: `state` owns the
//! buffer representation, `mutation` changes ring contents, `access` reads
//! ordered views, `stats` provides aggregate queries, and `display` handles
//! formatting.

mod access;
mod display;
mod mutation;
mod state;
mod stats;

pub use state::TokenRing;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let ring = TokenRing::new(10);
        assert_eq!(ring.capacity(), 10);
        assert!(ring.is_empty());
        assert_eq!(ring.remaining(), 10);
    }

    #[test]
    fn test_push_and_get() {
        let mut ring = TokenRing::new(5);
        ring.push(10);
        ring.push(20);
        ring.push(30);
        assert_eq!(ring.get(0), Some(10));
        assert_eq!(ring.get(2), Some(30));
        assert_eq!(ring.len(), 3);
    }

    #[test]
    fn test_overflow_evicts() {
        let mut ring = TokenRing::new(3);
        ring.push(1);
        ring.push(2);
        ring.push(3);
        let evicted = ring.push(4);
        assert_eq!(evicted, Some(1));
        assert_eq!(ring.to_vec(), vec![2, 3, 4]);
    }

    #[test]
    fn test_last() {
        let mut ring = TokenRing::new(5);
        ring.push(10);
        ring.push(20);
        assert_eq!(ring.last(), Some(20));
    }

    #[test]
    fn test_last_n() {
        let mut ring = TokenRing::new(5);
        ring.extend(&[1, 2, 3, 4, 5]);
        assert_eq!(ring.last_n(3), vec![3, 4, 5]);
        assert_eq!(ring.last_n(10), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_clear() {
        let mut ring = TokenRing::new(5);
        ring.extend(&[1, 2, 3]);
        ring.clear();
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn test_contains() {
        let mut ring = TokenRing::new(5);
        ring.extend(&[10, 20, 30]);
        assert!(ring.contains(20));
        assert!(!ring.contains(40));
    }

    #[test]
    fn test_count() {
        let mut ring = TokenRing::new(10);
        ring.extend(&[1, 2, 1, 3, 1]);
        assert_eq!(ring.count(1), 3);
        assert_eq!(ring.count(4), 0);
    }

    #[test]
    fn test_full_cycle() {
        let mut ring = TokenRing::new(3);
        for i in 0..10 {
            ring.push(i);
        }
        assert_eq!(ring.to_vec(), vec![7, 8, 9]);
        assert!(ring.is_full());
    }

    #[test]
    fn test_get_oob() {
        let ring = TokenRing::new(5);
        assert_eq!(ring.get(0), None);
    }

    #[test]
    fn test_empty_last() {
        let ring = TokenRing::new(5);
        assert_eq!(ring.last(), None);
    }

    #[test]
    fn test_display() {
        let mut ring = TokenRing::new(10);
        ring.extend(&[1, 2, 3]);
        assert_eq!(format!("{ring}"), "TokenRing(3/10)");
    }
}
