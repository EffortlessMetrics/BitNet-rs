//! Context window management.
//!
//! Track and manage the token context window for inference.

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

/// Context allocation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Fixed: prompt gets up to max_prompt, rest for generation.
    Fixed { max_prompt: usize },
    /// Dynamic: generation gets at least min_gen tokens.
    Dynamic { min_generation: usize },
    /// Even split between prompt and generation.
    EvenSplit,
}

/// Compute prompt and generation budgets.
pub fn compute_budgets(
    max_context: usize,
    prompt_len: usize,
    strategy: AllocationStrategy,
) -> (usize, usize) {
    match strategy {
        AllocationStrategy::Fixed { max_prompt } => {
            let prompt = prompt_len.min(max_prompt);
            let generation = max_context.saturating_sub(prompt);
            (prompt, generation)
        }
        AllocationStrategy::Dynamic { min_generation } => {
            let generation = min_generation.max(max_context.saturating_sub(prompt_len));
            let prompt = max_context.saturating_sub(generation);
            (prompt.min(prompt_len), generation)
        }
        AllocationStrategy::EvenSplit => {
            let half = max_context / 2;
            (prompt_len.min(half), half)
        }
    }
}

/// Context usage report.
#[derive(Debug, Clone)]
pub struct ContextReport {
    pub max_length: usize,
    pub used: usize,
    pub remaining: usize,
    pub utilization: f64,
}

impl ContextWindow {
    pub fn report(&self) -> ContextReport {
        ContextReport {
            max_length: self.max_length,
            used: self.current_length(),
            remaining: self.remaining(),
            utilization: self.utilization(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_window() {
        let w = ContextWindow::new(4096);
        assert_eq!(w.max_length(), 4096);
        assert!(w.is_empty());
        assert_eq!(w.remaining(), 4096);
    }

    #[test]
    fn test_append() {
        let mut w = ContextWindow::new(5);
        assert_eq!(w.append(&[1, 2, 3]), 3);
        assert_eq!(w.current_length(), 3);
        assert_eq!(w.remaining(), 2);
    }

    #[test]
    fn test_append_overflow() {
        let mut w = ContextWindow::new(3);
        assert_eq!(w.append(&[1, 2, 3, 4, 5]), 3);
        assert!(w.is_full());
    }

    #[test]
    fn test_last_n() {
        let mut w = ContextWindow::new(10);
        w.append(&[1, 2, 3, 4, 5]);
        assert_eq!(w.last_n(3), &[3, 4, 5]);
        assert_eq!(w.last_n(10), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_truncate() {
        let mut w = ContextWindow::new(10);
        w.append(&[1, 2, 3, 4, 5]);
        w.truncate_to_last(3);
        assert_eq!(w.tokens(), &[3, 4, 5]);
    }

    #[test]
    fn test_clear() {
        let mut w = ContextWindow::new(10);
        w.append(&[1, 2]);
        w.clear();
        assert!(w.is_empty());
    }

    #[test]
    fn test_can_fit() {
        let mut w = ContextWindow::new(5);
        w.append(&[1, 2, 3]);
        assert!(w.can_fit(2));
        assert!(!w.can_fit(3));
    }

    #[test]
    fn test_utilization() {
        let mut w = ContextWindow::new(4);
        w.append(&[1, 2]);
        assert!((w.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fixed_budget() {
        let (p, g) = compute_budgets(4096, 1000, AllocationStrategy::Fixed { max_prompt: 2048 });
        assert_eq!(p, 1000);
        assert_eq!(g, 3096);
    }

    #[test]
    fn test_dynamic_budget() {
        let (_p, g) =
            compute_budgets(4096, 3900, AllocationStrategy::Dynamic { min_generation: 256 });
        assert!(g >= 256);
    }

    #[test]
    fn test_even_split() {
        let (p, g) = compute_budgets(4096, 3000, AllocationStrategy::EvenSplit);
        assert_eq!(p, 2048);
        assert_eq!(g, 2048);
    }

    #[test]
    fn test_report() {
        let mut w = ContextWindow::new(100);
        w.append(&[1, 2, 3]);
        let r = w.report();
        assert_eq!(r.max_length, 100);
        assert_eq!(r.used, 3);
        assert_eq!(r.remaining, 97);
    }
}
