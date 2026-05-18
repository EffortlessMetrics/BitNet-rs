use super::TokenRing;

impl TokenRing {
    /// Get the i-th token (0 = oldest).
    pub fn get(&self, index: usize) -> Option<u32> {
        if index >= self.len {
            return None;
        }
        Some(self.buffer[self.physical_index(index)])
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
}
