//! KV cache management utilities.
//!
//! Memory-aware KV cache allocation and eviction.

/// KV cache configuration.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub dtype_bytes: usize, // 2 for f16, 4 for f32
}

impl KvCacheConfig {
    /// Bytes needed for one layer's K or V cache.
    pub fn per_layer_bytes(&self) -> usize {
        self.max_seq_len * self.num_kv_heads * self.head_dim * self.dtype_bytes
    }

    /// Total bytes for all layers (K + V).
    pub fn total_bytes(&self) -> usize {
        self.per_layer_bytes() * self.num_layers * 2
    }

    /// Bytes per token per layer (K + V).
    pub fn bytes_per_token(&self) -> usize {
        self.num_kv_heads * self.head_dim * self.dtype_bytes * 2
    }

    /// Total bytes per token across all layers.
    pub fn total_bytes_per_token(&self) -> usize {
        self.bytes_per_token() * self.num_layers
    }

    /// Max sequence length that fits in a byte budget.
    pub fn max_seq_for_budget(&self, budget_bytes: usize) -> usize {
        let per_token = self.total_bytes_per_token();
        if per_token == 0 {
            return 0;
        }
        budget_bytes / per_token
    }
}

/// Cache eviction strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvictionStrategy {
    None,
    SlidingWindow { window_size: usize },
    DropOldest { keep_recent: usize },
}

/// Manages KV cache state.
#[derive(Debug)]
pub struct KvCacheManager {
    config: KvCacheConfig,
    current_len: usize,
    eviction: EvictionStrategy,
    eviction_count: usize,
}

impl KvCacheManager {
    pub fn new(config: KvCacheConfig, eviction: EvictionStrategy) -> Self {
        Self { config, current_len: 0, eviction, eviction_count: 0 }
    }

    /// Current sequence length in cache.
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Remaining capacity in tokens.
    pub fn remaining(&self) -> usize {
        self.config.max_seq_len.saturating_sub(self.current_len)
    }

    /// Is cache full?
    pub fn is_full(&self) -> bool {
        self.current_len >= self.config.max_seq_len
    }

    /// Current memory usage in bytes.
    pub fn memory_used(&self) -> usize {
        self.current_len * self.config.total_bytes_per_token()
    }

    /// Append tokens to cache.
    pub fn append(&mut self, num_tokens: usize) -> AppendResult {
        // For sliding window, enforce the window constraint proactively.
        if let EvictionStrategy::SlidingWindow { window_size } = self.eviction
            && self.current_len + num_tokens > window_size
        {
            let target_len = window_size.saturating_sub(num_tokens);
            let evicted = self.current_len.saturating_sub(target_len);
            self.current_len = target_len + num_tokens;
            self.eviction_count += evicted;
            return AppendResult::Evicted { evicted, new_len: self.current_len };
        }

        let available = self.remaining();
        if num_tokens <= available {
            self.current_len += num_tokens;
            return AppendResult::Ok { new_len: self.current_len };
        }

        // Need eviction
        match self.eviction {
            EvictionStrategy::None => AppendResult::Full { needed: num_tokens, available },
            EvictionStrategy::SlidingWindow { .. } => {
                // Already handled above
                unreachable!()
            }
            EvictionStrategy::DropOldest { keep_recent } => {
                let target = keep_recent.min(self.current_len);
                let evicted = self.current_len - target;
                self.current_len = target + num_tokens;
                self.eviction_count += evicted;
                if self.current_len > self.config.max_seq_len {
                    self.current_len = self.config.max_seq_len;
                }
                AppendResult::Evicted { evicted, new_len: self.current_len }
            }
        }
    }

    /// Reset cache state.
    pub fn clear(&mut self) {
        self.current_len = 0;
    }

    pub fn eviction_count(&self) -> usize {
        self.eviction_count
    }

    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }
}

/// Result of appending tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum AppendResult {
    Ok { new_len: usize },
    Evicted { evicted: usize, new_len: usize },
    Full { needed: usize, available: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KvCacheConfig {
        KvCacheConfig {
            num_layers: 4,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 100,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn test_per_layer_bytes() {
        let c = test_config();
        assert_eq!(c.per_layer_bytes(), 100 * 4 * 64 * 2);
    }

    #[test]
    fn test_total_bytes() {
        let c = test_config();
        let expected = c.per_layer_bytes() * 4 * 2; // 4 layers, K+V
        assert_eq!(c.total_bytes(), expected);
    }

    #[test]
    fn test_bytes_per_token() {
        let c = test_config();
        assert_eq!(c.bytes_per_token(), 4 * 64 * 2 * 2); // kv_heads * head_dim * dtype * 2
    }

    #[test]
    fn test_max_seq_for_budget() {
        let c = test_config();
        let budget = c.total_bytes_per_token() * 50;
        assert_eq!(c.max_seq_for_budget(budget), 50);
    }

    #[test]
    fn test_new_manager() {
        let m = KvCacheManager::new(test_config(), EvictionStrategy::None);
        assert_eq!(m.current_len(), 0);
        assert_eq!(m.remaining(), 100);
        assert!(!m.is_full());
    }

    #[test]
    fn test_append_ok() {
        let mut m = KvCacheManager::new(test_config(), EvictionStrategy::None);
        let r = m.append(10);
        assert_eq!(r, AppendResult::Ok { new_len: 10 });
        assert_eq!(m.current_len(), 10);
    }

    #[test]
    fn test_append_full() {
        let mut m = KvCacheManager::new(test_config(), EvictionStrategy::None);
        m.append(100);
        let r = m.append(5);
        assert!(matches!(r, AppendResult::Full { .. }));
    }

    #[test]
    fn test_sliding_window() {
        let mut m =
            KvCacheManager::new(test_config(), EvictionStrategy::SlidingWindow { window_size: 50 });
        m.append(50); // fill to 50
        let r = m.append(10); // should evict 10 to make room
        assert!(matches!(r, AppendResult::Evicted { .. }));
        assert_eq!(m.current_len(), 50); // window maintained
    }

    #[test]
    fn test_drop_oldest() {
        let mut m =
            KvCacheManager::new(test_config(), EvictionStrategy::DropOldest { keep_recent: 80 });
        m.append(100); // fill completely
        let r = m.append(10); // evict oldest, keep 80, add 10
        assert!(matches!(r, AppendResult::Evicted { .. }));
        assert!(m.current_len() <= 100);
    }

    #[test]
    fn test_clear() {
        let mut m = KvCacheManager::new(test_config(), EvictionStrategy::None);
        m.append(50);
        m.clear();
        assert_eq!(m.current_len(), 0);
    }

    #[test]
    fn test_memory_used() {
        let mut m = KvCacheManager::new(test_config(), EvictionStrategy::None);
        m.append(10);
        assert_eq!(m.memory_used(), 10 * test_config().total_bytes_per_token());
    }

    #[test]
    fn test_eviction_count() {
        let mut m =
            KvCacheManager::new(test_config(), EvictionStrategy::SlidingWindow { window_size: 20 });
        m.append(20);
        m.append(5);
        assert!(m.eviction_count() > 0);
    }

    #[test]
    fn test_phi4_config() {
        // Phi-4: 40 layers, 10 kv heads, head_dim=128, max=16384, f16
        let c = KvCacheConfig {
            num_layers: 40,
            num_kv_heads: 10,
            head_dim: 128,
            max_seq_len: 16384,
            dtype_bytes: 2,
        };
        let total = c.total_bytes();
        // Expected: ~3.2 GB
        assert!(total > 3_000_000_000);
        assert!(total < 4_000_000_000);
    }
}
