use super::TokenRing;

impl TokenRing {
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
        for &token in tokens {
            self.push(token);
        }
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}
