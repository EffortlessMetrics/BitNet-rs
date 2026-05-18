use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── Rate limiter (simple token-bucket) ──────────────────────────────────────

/// Simple per-key request counter for rate limiting.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Maximum requests per window.
    pub limit: u32,
    /// Window duration.
    pub window: Duration,
    counters: HashMap<String, (u32, Instant)>,
}

impl RateLimiter {
    pub fn new(limit: u32, window: Duration) -> Self {
        Self { limit, window, counters: HashMap::new() }
    }

    /// Check whether `key` is allowed. Returns `true` if under limit.
    pub fn check(&mut self, key: &str) -> bool {
        let now = Instant::now();
        let entry = self.counters.entry(key.to_string()).or_insert((0, now));
        if now.duration_since(entry.1) >= self.window {
            *entry = (1, now);
            true
        } else if entry.0 < self.limit {
            entry.0 += 1;
            true
        } else {
            false
        }
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.counters.clear();
    }
}
