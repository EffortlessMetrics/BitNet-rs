/// Context window state.
#[derive(Debug, Clone)]
pub struct ContextWindow {
    max_length: usize,
    tokens: Vec<u32>,
}

impl ContextWindow {
    pub fn new(max_length: usize) -> Self {
        Self { max_length, tokens: Vec::new() }
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    pub fn current_length(&self) -> usize {
        self.tokens.len()
    }

    pub fn remaining(&self) -> usize {
        self.max_length.saturating_sub(self.tokens.len())
    }

    pub fn is_full(&self) -> bool {
        self.tokens.len() >= self.max_length
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn utilization(&self) -> f64 {
        if self.max_length == 0 {
            return 0.0;
        }
        self.tokens.len() as f64 / self.max_length as f64
    }

    /// Append tokens, returns how many were actually added.
    pub fn append(&mut self, tokens: &[u32]) -> usize {
        let space = self.remaining();
        let to_add = tokens.len().min(space);
        self.tokens.extend_from_slice(&tokens[..to_add]);
        to_add
    }

    /// Get all tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Get last N tokens.
    pub fn last_n(&self, n: usize) -> &[u32] {
        let start = self.tokens.len().saturating_sub(n);
        &self.tokens[start..]
    }

    /// Truncate to keep only the last N tokens (sliding window).
    pub fn truncate_to_last(&mut self, n: usize) {
        if self.tokens.len() > n {
            let start = self.tokens.len() - n;
            self.tokens = self.tokens[start..].to_vec();
        }
    }

    /// Clear the context.
    pub fn clear(&mut self) {
        self.tokens.clear();
    }

    /// Check if a given number of new tokens would fit.
    pub fn can_fit(&self, count: usize) -> bool {
        self.remaining() >= count
    }
}
