use super::TokenRing;

impl TokenRing {
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
