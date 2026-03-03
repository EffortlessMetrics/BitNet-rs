//! Fixed-capacity ring buffer for token IDs.
//!
//! Efficient circular buffer for streaming token storage with O(1)
//! push/pop and sliding window semantics.

/// A fixed-capacity ring buffer for token IDs.
#[derive(Debug, Clone)]
pub struct TokenRing {
    buffer: Vec<u32>,
    capacity: usize,
    head: usize, // next write position
    len: usize,  // current number of items
}

impl TokenRing {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring buffer capacity must be > 0");
        Self { buffer: vec![0; capacity], capacity, head: 0, len: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Push a token. If full, overwrites oldest.
    pub fn push(&mut self, token: u32) -> Option<u32> {
        let evicted = if self.is_full() {
            Some(self.buffer[self.head])
        } else {
            self.len += 1;
            None
        };
        self.buffer[self.head] = token;
        self.head = (self.head + 1) % self.capacity;
        evicted
    }

    /// Push multiple tokens.
    pub fn extend(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.push(t);
        }
    }

    /// Get the i-th token (0 = oldest).
    pub fn get(&self, index: usize) -> Option<u32> {
        if index >= self.len {
            return None;
        }
        let start = if self.len < self.capacity { 0 } else { self.head };
        let pos = (start + index) % self.capacity;
        Some(self.buffer[pos])
    }

    /// Most recently pushed token.
    pub fn last(&self) -> Option<u32> {
        if self.is_empty() {
            return None;
        }
        let pos = (self.head + self.capacity - 1) % self.capacity;
        Some(self.buffer[pos])
    }

    /// Get the last N tokens (most recent).
    pub fn last_n(&self, n: usize) -> Vec<u32> {
        let n = n.min(self.len);
        let start = self.len.saturating_sub(n);
        (start..self.len).filter_map(|i| self.get(i)).collect()
    }

    /// Collect all tokens in order (oldest first).
    pub fn to_vec(&self) -> Vec<u32> {
        (0..self.len).filter_map(|i| self.get(i)).collect()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Check if a token exists in the buffer.
    pub fn contains(&self, token: u32) -> bool {
        self.to_vec().contains(&token)
    }

    /// Count occurrences of a token.
    pub fn count(&self, token: u32) -> usize {
        self.to_vec().iter().filter(|&&t| t == token).count()
    }

    /// Remaining capacity before eviction starts.
    pub fn remaining(&self) -> usize {
        self.capacity - self.len
    }
}

impl std::fmt::Display for TokenRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenRing({}/{})", self.len, self.capacity)
    }
}

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
