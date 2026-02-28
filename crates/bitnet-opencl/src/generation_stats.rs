//! Detailed timing and throughput statistics for text generation.
//!
//! [`GenerationStats`] captures per-phase timing (prefill vs decode),
//! throughput, and memory usage.  [`StatsCollector`] accumulates
//! per-token timestamps during generation.

use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Detailed statistics collected during a generation run.
///
/// Fields are populated by [`StatsCollector`] and can be serialized to
/// JSON for receipt / telemetry pipelines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    /// Wall-clock time for the entire generation (prefill + decode).
    pub total_time_ms: f64,
    /// Time spent processing the input prompt (parallel phase).
    pub prefill_time_ms: f64,
    /// Time spent in autoregressive token generation.
    pub decode_time_ms: f64,
    /// Number of tokens produced (excluding prompt tokens).
    pub tokens_generated: usize,
    /// Decode throughput: `tokens_generated / decode_time_s`.
    pub tokens_per_second: f64,
    /// Latency from start until the first generated token is ready.
    pub first_token_latency_ms: f64,
    /// Peak process memory during generation (0 when unavailable).
    pub peak_memory_bytes: u64,
}

impl Default for GenerationStats {
    fn default() -> Self {
        Self {
            total_time_ms: 0.0,
            prefill_time_ms: 0.0,
            decode_time_ms: 0.0,
            tokens_generated: 0,
            tokens_per_second: 0.0,
            first_token_latency_ms: 0.0,
            peak_memory_bytes: 0,
        }
    }
}

impl GenerationStats {
    /// Human-readable multi-line summary.
    #[must_use]
    pub fn format_stats(&self) -> String {
        format!(
            "Generation stats:\n  \
             tokens generated : {}\n  \
             total time       : {:.1} ms\n  \
             prefill time     : {:.1} ms\n  \
             decode time      : {:.1} ms\n  \
             throughput       : {:.2} tok/s\n  \
             first-token lat  : {:.1} ms\n  \
             peak memory      : {} bytes",
            self.tokens_generated,
            self.total_time_ms,
            self.prefill_time_ms,
            self.decode_time_ms,
            self.tokens_per_second,
            self.first_token_latency_ms,
            self.peak_memory_bytes,
        )
    }
}

impl fmt::Display for GenerationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} tokens in {:.1}ms ({:.2} tok/s)",
            self.tokens_generated, self.total_time_ms, self.tokens_per_second,
        )
    }
}

// ---------------------------------------------------------------------------
// Collector
// ---------------------------------------------------------------------------

/// Accumulates timing data during a generation run.
///
/// Call [`begin_prefill`](Self::begin_prefill) /
/// [`end_prefill`](Self::end_prefill), then
/// [`record_token`](Self::record_token) for every decode step, then
/// [`finish`](Self::finish) to produce the final [`GenerationStats`].
pub struct StatsCollector {
    start: Instant,
    prefill_end: Option<Instant>,
    token_times: Vec<Instant>,
}

impl StatsCollector {
    /// Create a new collector; records the start timestamp immediately.
    #[must_use]
    pub fn new() -> Self {
        Self { start: Instant::now(), prefill_end: None, token_times: Vec::new() }
    }

    /// Mark the start of the prefill phase (resets the start clock).
    pub fn begin_prefill(&mut self) {
        self.start = Instant::now();
    }

    /// Mark the end of the prefill phase.
    pub fn end_prefill(&mut self) {
        self.prefill_end = Some(Instant::now());
    }

    /// Record that a token was just produced.
    pub fn record_token(&mut self) {
        self.token_times.push(Instant::now());
    }

    /// Finalize and return the stats snapshot.
    #[must_use]
    pub fn finish(&self) -> GenerationStats {
        let now = Instant::now();
        let total = now.duration_since(self.start).as_secs_f64() * 1000.0;

        let prefill_ms =
            self.prefill_end.map_or(0.0, |t| t.duration_since(self.start).as_secs_f64() * 1000.0);

        let decode_start = self.prefill_end.unwrap_or(self.start);
        let decode_ms = now.duration_since(decode_start).as_secs_f64() * 1000.0;

        let tokens_generated = self.token_times.len();

        let first_token_latency_ms = self
            .token_times
            .first()
            .map_or(0.0, |t| t.duration_since(self.start).as_secs_f64() * 1000.0);

        let decode_secs = decode_ms / 1000.0;
        #[allow(clippy::cast_precision_loss)]
        let tokens_per_second =
            if decode_secs > 0.0 { tokens_generated as f64 / decode_secs } else { 0.0 };

        GenerationStats {
            total_time_ms: total,
            prefill_time_ms: prefill_ms,
            decode_time_ms: decode_ms,
            tokens_generated,
            tokens_per_second,
            first_token_latency_ms,
            peak_memory_bytes: 0,
        }
    }
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stats_are_zero() {
        let s = GenerationStats::default();
        assert_eq!(s.tokens_generated, 0);
        assert_eq!(s.tokens_per_second, 0.0);
        assert_eq!(s.peak_memory_bytes, 0);
    }

    #[test]
    fn format_stats_contains_key_labels() {
        let s = GenerationStats {
            total_time_ms: 100.0,
            prefill_time_ms: 20.0,
            decode_time_ms: 80.0,
            tokens_generated: 10,
            tokens_per_second: 125.0,
            first_token_latency_ms: 25.0,
            peak_memory_bytes: 1024,
        };
        let text = s.format_stats();
        assert!(text.contains("tokens generated"));
        assert!(text.contains("125.00 tok/s"));
        assert!(text.contains("1024 bytes"));
    }

    #[test]
    fn display_impl_shows_summary() {
        let s = GenerationStats {
            tokens_generated: 5,
            total_time_ms: 50.0,
            tokens_per_second: 100.0,
            ..Default::default()
        };
        let text = format!("{s}");
        assert!(text.contains("5 tokens"));
        assert!(text.contains("100.00 tok/s"));
    }

    #[test]
    fn stats_round_trips_through_json() {
        let s = GenerationStats {
            total_time_ms: 99.5,
            prefill_time_ms: 10.0,
            decode_time_ms: 89.5,
            tokens_generated: 7,
            tokens_per_second: 78.21,
            first_token_latency_ms: 12.3,
            peak_memory_bytes: 4096,
        };
        let json = serde_json::to_string(&s).unwrap();
        let de: GenerationStats = serde_json::from_str(&json).unwrap();
        assert_eq!(de.tokens_generated, 7);
        assert!((de.tokens_per_second - 78.21).abs() < 0.01);
    }

    #[test]
    fn collector_records_tokens() {
        let mut c = StatsCollector::new();
        c.begin_prefill();
        c.end_prefill();
        c.record_token();
        c.record_token();
        c.record_token();
        let stats = c.finish();
        assert_eq!(stats.tokens_generated, 3);
        assert!(stats.total_time_ms >= 0.0);
        assert!(stats.prefill_time_ms >= 0.0);
        assert!(stats.decode_time_ms >= 0.0);
    }

    #[test]
    fn collector_without_prefill_still_works() {
        let mut c = StatsCollector::new();
        c.record_token();
        let stats = c.finish();
        assert_eq!(stats.tokens_generated, 1);
        assert_eq!(stats.prefill_time_ms, 0.0);
    }

    #[test]
    fn collector_no_tokens_yields_zero_throughput() {
        let c = StatsCollector::new();
        let stats = c.finish();
        assert_eq!(stats.tokens_generated, 0);
        assert_eq!(stats.first_token_latency_ms, 0.0);
    }

    #[test]
    fn collector_default_matches_new() {
        let a = StatsCollector::new().finish();
        let b = StatsCollector::default().finish();
        assert_eq!(a.tokens_generated, b.tokens_generated);
    }
}
