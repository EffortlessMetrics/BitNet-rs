//! Context window management for inference.
//!
//! Tracks token positions within the model's context window,
//! handles overflow, and manages prompt/generation boundaries.

/// Context window configuration.
#[derive(Debug, Clone)]
pub struct ContextWindow {
    /// Maximum context length (e.g., 4096, 8192, 16384).
    pub max_length: usize,
    /// Current position in the context.
    pub position: usize,
    /// Position where prompt ends and generation begins.
    pub prompt_end: usize,
}

impl ContextWindow {
    pub fn new(max_length: usize) -> Self {
        Self { max_length, position: 0, prompt_end: 0 }
    }

    /// Advance position by n tokens.
    pub fn advance(&mut self, n: usize) {
        self.position += n;
    }

    /// Set the prompt boundary at current position.
    pub fn mark_prompt_end(&mut self) {
        self.prompt_end = self.position;
    }

    /// Remaining capacity in the context window.
    pub fn remaining(&self) -> usize {
        self.max_length.saturating_sub(self.position)
    }

    /// Whether the context window is full.
    pub fn is_full(&self) -> bool {
        self.position >= self.max_length
    }

    /// Tokens generated so far (after prompt).
    pub fn generated_tokens(&self) -> usize {
        self.position.saturating_sub(self.prompt_end)
    }

    /// Prompt length in tokens.
    pub fn prompt_length(&self) -> usize {
        self.prompt_end
    }

    /// Utilization as a fraction (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.max_length == 0 {
            return 0.0;
        }
        self.position as f64 / self.max_length as f64
    }

    /// Reset the context window.
    pub fn reset(&mut self) {
        self.position = 0;
        self.prompt_end = 0;
    }

    /// Whether we can fit n more tokens.
    pub fn can_fit(&self, n: usize) -> bool {
        self.position + n <= self.max_length
    }

    /// Truncate to fit within max_length (drop oldest tokens).
    pub fn truncate_to_fit(&mut self) {
        if self.position > self.max_length {
            let overflow = self.position - self.max_length;
            self.position = self.max_length;
            self.prompt_end = self.prompt_end.saturating_sub(overflow);
        }
    }
}

/// Common context window sizes.
pub fn context_2k() -> ContextWindow {
    ContextWindow::new(2048)
}
pub fn context_4k() -> ContextWindow {
    ContextWindow::new(4096)
}
pub fn context_8k() -> ContextWindow {
    ContextWindow::new(8192)
}
pub fn context_16k() -> ContextWindow {
    ContextWindow::new(16384)
}
pub fn context_32k() -> ContextWindow {
    ContextWindow::new(32768)
}

/// Estimate KV cache memory for a context window.
pub fn kv_cache_bytes(
    ctx_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    bytes_per_element: usize,
) -> u64 {
    // 2 (K+V) * layers * kv_heads * head_dim * ctx_len * bytes
    2 * num_layers as u64
        * num_kv_heads as u64
        * head_dim as u64
        * ctx_len as u64
        * bytes_per_element as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let cw = ContextWindow::new(4096);
        assert_eq!(cw.max_length, 4096);
        assert_eq!(cw.position, 0);
    }

    #[test]
    fn test_advance() {
        let mut cw = ContextWindow::new(100);
        cw.advance(10);
        assert_eq!(cw.position, 10);
        assert_eq!(cw.remaining(), 90);
    }

    #[test]
    fn test_mark_prompt_end() {
        let mut cw = ContextWindow::new(100);
        cw.advance(20);
        cw.mark_prompt_end();
        assert_eq!(cw.prompt_length(), 20);
        cw.advance(5);
        assert_eq!(cw.generated_tokens(), 5);
    }

    #[test]
    fn test_is_full() {
        let mut cw = ContextWindow::new(10);
        assert!(!cw.is_full());
        cw.advance(10);
        assert!(cw.is_full());
    }

    #[test]
    fn test_remaining() {
        let mut cw = ContextWindow::new(100);
        cw.advance(75);
        assert_eq!(cw.remaining(), 25);
    }

    #[test]
    fn test_utilization() {
        let mut cw = ContextWindow::new(100);
        cw.advance(50);
        assert!((cw.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_utilization_zero() {
        let cw = ContextWindow::new(0);
        assert_eq!(cw.utilization(), 0.0);
    }

    #[test]
    fn test_reset() {
        let mut cw = ContextWindow::new(100);
        cw.advance(50);
        cw.mark_prompt_end();
        cw.reset();
        assert_eq!(cw.position, 0);
        assert_eq!(cw.prompt_end, 0);
    }

    #[test]
    fn test_can_fit() {
        let mut cw = ContextWindow::new(10);
        cw.advance(8);
        assert!(cw.can_fit(2));
        assert!(!cw.can_fit(3));
    }

    #[test]
    fn test_truncate() {
        let mut cw = ContextWindow::new(100);
        cw.advance(150);
        cw.prompt_end = 50;
        cw.truncate_to_fit();
        assert_eq!(cw.position, 100);
        assert_eq!(cw.prompt_end, 0);
    }

    #[test]
    fn test_truncate_no_overflow() {
        let mut cw = ContextWindow::new(100);
        cw.advance(50);
        cw.truncate_to_fit();
        assert_eq!(cw.position, 50);
    }

    #[test]
    fn test_context_presets() {
        assert_eq!(context_4k().max_length, 4096);
        assert_eq!(context_16k().max_length, 16384);
        assert_eq!(context_32k().max_length, 32768);
    }

    #[test]
    fn test_kv_cache_bytes() {
        // Phi-4: 40 layers, 10 kv heads, 128 head dim, fp16
        let bytes = kv_cache_bytes(4096, 40, 10, 128, 2);
        assert!(bytes > 0);
        // Expected: 2 * 40 * 10 * 128 * 4096 * 2 = 838,860,800
        assert_eq!(bytes, 838_860_800);
    }

    #[test]
    fn test_kv_cache_scales() {
        let short = kv_cache_bytes(1024, 32, 8, 128, 2);
        let long = kv_cache_bytes(4096, 32, 8, 128, 2);
        assert_eq!(long, short * 4);
    }

    #[test]
    fn test_generated_no_prompt() {
        let cw = ContextWindow::new(100);
        assert_eq!(cw.generated_tokens(), 0);
    }
}
