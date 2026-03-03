//! Sliding window attention configuration.
//!
//! Supports full attention, fixed sliding window, and hybrid (local + global)
//! patterns used by models like Mistral, Gemma-2, and Longformer.

/// Sliding window attention mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    /// Full quadratic attention (no window).
    Full,
    /// Fixed-size sliding window.
    Sliding { window_size: usize },
    /// Hybrid: alternating full and sliding layers.
    Hybrid { window_size: usize, full_every_n: usize },
}

/// Configuration for sliding window attention.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    pub mode: WindowMode,
    pub causal: bool,
    pub left_padding: usize,
}

impl SlidingWindowConfig {
    pub fn full() -> Self {
        Self { mode: WindowMode::Full, causal: true, left_padding: 0 }
    }

    pub fn sliding(window_size: usize) -> Self {
        Self { mode: WindowMode::Sliding { window_size }, causal: true, left_padding: 0 }
    }

    pub fn hybrid(window_size: usize, full_every_n: usize) -> Self {
        Self {
            mode: WindowMode::Hybrid { window_size, full_every_n },
            causal: true,
            left_padding: 0,
        }
    }

    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    pub fn with_left_padding(mut self, padding: usize) -> Self {
        self.left_padding = padding;
        self
    }

    /// Check if a layer should use full attention.
    pub fn is_full_attention_layer(&self, layer_idx: usize) -> bool {
        match self.mode {
            WindowMode::Full => true,
            WindowMode::Sliding { .. } => false,
            WindowMode::Hybrid { full_every_n, .. } => {
                full_every_n > 0 && (layer_idx % full_every_n == 0)
            }
        }
    }

    /// Get the effective window size for a layer.
    pub fn effective_window(&self, layer_idx: usize, seq_len: usize) -> usize {
        match self.mode {
            WindowMode::Full => seq_len,
            WindowMode::Sliding { window_size } => window_size.min(seq_len),
            WindowMode::Hybrid { window_size, full_every_n } => {
                if full_every_n > 0 && (layer_idx % full_every_n == 0) {
                    seq_len
                } else {
                    window_size.min(seq_len)
                }
            }
        }
    }

    /// Compute attention mask start position for a query at position `q_pos`.
    pub fn mask_start(&self, layer_idx: usize, q_pos: usize, seq_len: usize) -> usize {
        let window = self.effective_window(layer_idx, seq_len);
        if q_pos >= window { q_pos - window + 1 } else { 0 }
    }

    /// Compute attention mask end position for a query at position `q_pos`.
    pub fn mask_end(&self, q_pos: usize, seq_len: usize) -> usize {
        if self.causal { (q_pos + 1).min(seq_len) } else { seq_len }
    }

    /// Estimate KV cache memory for this config (in tokens per layer).
    pub fn kv_cache_tokens_per_layer(&self, seq_len: usize, layer_idx: usize) -> usize {
        self.effective_window(layer_idx, seq_len)
    }

    /// Total KV cache tokens across all layers.
    pub fn total_kv_cache_tokens(&self, seq_len: usize, num_layers: usize) -> usize {
        (0..num_layers).map(|l| self.kv_cache_tokens_per_layer(seq_len, l)).sum()
    }

    /// Memory savings ratio vs full attention (0.0 = no savings, approaching 1.0 = large savings).
    pub fn memory_savings_ratio(&self, seq_len: usize, num_layers: usize) -> f64 {
        if seq_len == 0 || num_layers == 0 {
            return 0.0;
        }
        let full = seq_len * num_layers;
        let actual = self.total_kv_cache_tokens(seq_len, num_layers);
        1.0 - (actual as f64 / full as f64)
    }
}

/// Create a Mistral-style config (4096 window, all sliding).
pub fn mistral_config() -> SlidingWindowConfig {
    SlidingWindowConfig::sliding(4096)
}

/// Create a Gemma-2 style config (4096 window, full every 2nd layer).
pub fn gemma2_config() -> SlidingWindowConfig {
    SlidingWindowConfig::hybrid(4096, 2)
}

/// Create a full attention config (used by most models like Phi-4, LLaMA).
pub fn full_attention_config() -> SlidingWindowConfig {
    SlidingWindowConfig::full()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_attention() {
        let cfg = SlidingWindowConfig::full();
        assert!(cfg.is_full_attention_layer(0));
        assert_eq!(cfg.effective_window(0, 1024), 1024);
    }

    #[test]
    fn test_sliding_window() {
        let cfg = SlidingWindowConfig::sliding(512);
        assert!(!cfg.is_full_attention_layer(0));
        assert_eq!(cfg.effective_window(0, 1024), 512);
        assert_eq!(cfg.effective_window(0, 256), 256); // clamp to seq_len
    }

    #[test]
    fn test_hybrid() {
        let cfg = SlidingWindowConfig::hybrid(512, 2);
        assert!(cfg.is_full_attention_layer(0));
        assert!(!cfg.is_full_attention_layer(1));
        assert!(cfg.is_full_attention_layer(2));
        assert_eq!(cfg.effective_window(0, 1024), 1024); // full
        assert_eq!(cfg.effective_window(1, 1024), 512); // sliding
    }

    #[test]
    fn test_mask_start_sliding() {
        let cfg = SlidingWindowConfig::sliding(4);
        assert_eq!(cfg.mask_start(0, 0, 10), 0);
        assert_eq!(cfg.mask_start(0, 3, 10), 0);
        assert_eq!(cfg.mask_start(0, 5, 10), 2);
    }

    #[test]
    fn test_mask_end_causal() {
        let cfg = SlidingWindowConfig::full();
        assert_eq!(cfg.mask_end(3, 10), 4);
        assert_eq!(cfg.mask_end(9, 10), 10);
    }

    #[test]
    fn test_mask_end_non_causal() {
        let cfg = SlidingWindowConfig::full().with_causal(false);
        assert_eq!(cfg.mask_end(3, 10), 10);
    }

    #[test]
    fn test_kv_cache_tokens() {
        let cfg = SlidingWindowConfig::sliding(512);
        assert_eq!(cfg.kv_cache_tokens_per_layer(1024, 0), 512);
    }

    #[test]
    fn test_total_kv_cache() {
        let cfg = SlidingWindowConfig::sliding(100);
        let total = cfg.total_kv_cache_tokens(1000, 10);
        assert_eq!(total, 1000); // 100 * 10
    }

    #[test]
    fn test_memory_savings() {
        let cfg = SlidingWindowConfig::sliding(100);
        let ratio = cfg.memory_savings_ratio(1000, 10);
        assert!((ratio - 0.9).abs() < 0.01); // 90% savings
    }

    #[test]
    fn test_memory_savings_full() {
        let ratio = SlidingWindowConfig::full().memory_savings_ratio(1000, 10);
        assert!(ratio.abs() < 0.01); // no savings
    }

    #[test]
    fn test_presets() {
        let m = mistral_config();
        assert_eq!(m.effective_window(0, 8192), 4096);
        let g = gemma2_config();
        assert!(g.is_full_attention_layer(0));
        assert!(!g.is_full_attention_layer(1));
        let f = full_attention_config();
        assert!(f.is_full_attention_layer(42));
    }

    #[test]
    fn test_builder() {
        let cfg = SlidingWindowConfig::sliding(256).with_causal(false).with_left_padding(10);
        assert!(!cfg.causal);
        assert_eq!(cfg.left_padding, 10);
    }
}
