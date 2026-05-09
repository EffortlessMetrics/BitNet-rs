//! Lightweight performance profiler.
//!
//! Hierarchical timing spans for profiling inference stages.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A recorded timing span.
#[derive(Debug, Clone)]
pub struct Span {
    pub name: String,
    pub duration: Duration,
    pub parent: Option<String>,
}

/// Accumulated stats for a named region.
#[derive(Debug, Clone)]
pub struct RegionStats {
    pub name: String,
    pub call_count: u64,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl RegionStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            call_count: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
        }
    }

    fn record(&mut self, d: Duration) {
        self.call_count += 1;
        self.total_time += d;
        self.min_time = self.min_time.min(d);
        self.max_time = self.max_time.max(d);
    }

    pub fn avg_time(&self) -> Duration {
        if self.call_count == 0 {
            return Duration::ZERO;
        }
        self.total_time / self.call_count as u32
    }
}

/// Active span guard.
pub struct SpanGuard<'a> {
    profiler: &'a mut Profiler,
    name: String,
    start: Instant,
}

impl<'a> SpanGuard<'a> {
    pub fn finish(self) -> Duration {
        let d = self.start.elapsed();
        self.profiler.record_span(&self.name, d);
        d
    }
}

/// Performance profiler.
#[derive(Debug)]
pub struct Profiler {
    regions: HashMap<String, RegionStats>,
    spans: Vec<Span>,
    enabled: bool,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self { regions: HashMap::new(), spans: Vec::new(), enabled: true }
    }

    pub fn disabled() -> Self {
        Self { regions: HashMap::new(), spans: Vec::new(), enabled: false }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start a timed span (manual finish).
    pub fn start_span(&mut self, name: impl Into<String>) -> SpanGuard<'_> {
        SpanGuard { profiler: self, name: name.into(), start: Instant::now() }
    }

    /// Record a pre-timed span.
    pub fn record_span(&mut self, name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }
        self.spans.push(Span { name: name.to_string(), duration, parent: None });
        self.regions
            .entry(name.to_string())
            .or_insert_with(|| RegionStats::new(name))
            .record(duration);
    }

    /// Time a closure.
    pub fn time<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let d = start.elapsed();
        self.record_span(name, d);
        result
    }

    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    pub fn span_count(&self) -> usize {
        self.spans.len()
    }

    pub fn get_region(&self, name: &str) -> Option<&RegionStats> {
        self.regions.get(name)
    }

    pub fn total_time(&self) -> Duration {
        self.regions.values().map(|r| r.total_time).sum()
    }

    /// Get all regions sorted by total time (descending).
    pub fn sorted_regions(&self) -> Vec<&RegionStats> {
        let mut regions: Vec<_> = self.regions.values().collect();
        regions.sort_by_key(|a| std::cmp::Reverse(a.total_time));
        regions
    }

    /// Clear all data.
    pub fn reset(&mut self) {
        self.regions.clear();
        self.spans.clear();
    }

    /// Merge another profiler's data.
    pub fn merge(&mut self, other: &Profiler) {
        for (name, stats) in &other.regions {
            let entry = self.regions.entry(name.clone()).or_insert_with(|| RegionStats::new(name));
            entry.call_count += stats.call_count;
            entry.total_time += stats.total_time;
            entry.min_time = entry.min_time.min(stats.min_time);
            entry.max_time = entry.max_time.max(stats.max_time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_span() {
        let mut p = Profiler::new();
        p.record_span("matmul", Duration::from_millis(10));
        p.record_span("matmul", Duration::from_millis(20));
        let r = p.get_region("matmul").unwrap();
        assert_eq!(r.call_count, 2);
        assert_eq!(r.total_time, Duration::from_millis(30));
    }

    #[test]
    fn test_avg_time() {
        let mut p = Profiler::new();
        p.record_span("op", Duration::from_millis(10));
        p.record_span("op", Duration::from_millis(30));
        assert_eq!(p.get_region("op").unwrap().avg_time(), Duration::from_millis(20));
    }

    #[test]
    fn test_min_max() {
        let mut p = Profiler::new();
        p.record_span("op", Duration::from_millis(5));
        p.record_span("op", Duration::from_millis(15));
        let r = p.get_region("op").unwrap();
        assert_eq!(r.min_time, Duration::from_millis(5));
        assert_eq!(r.max_time, Duration::from_millis(15));
    }

    #[test]
    fn test_time_closure() {
        let mut p = Profiler::new();
        let result = p.time("add", || 2 + 3);
        assert_eq!(result, 5);
        assert_eq!(p.get_region("add").unwrap().call_count, 1);
    }

    #[test]
    fn test_disabled() {
        let mut p = Profiler::disabled();
        p.record_span("op", Duration::from_millis(10));
        assert_eq!(p.region_count(), 0);
    }

    #[test]
    fn test_sorted_regions() {
        let mut p = Profiler::new();
        p.record_span("fast", Duration::from_millis(1));
        p.record_span("slow", Duration::from_millis(100));
        let sorted = p.sorted_regions();
        assert_eq!(sorted[0].name, "slow");
    }

    #[test]
    fn test_reset() {
        let mut p = Profiler::new();
        p.record_span("op", Duration::from_millis(10));
        p.reset();
        assert_eq!(p.region_count(), 0);
        assert_eq!(p.span_count(), 0);
    }

    #[test]
    fn test_merge() {
        let mut a = Profiler::new();
        a.record_span("op", Duration::from_millis(10));
        let mut b = Profiler::new();
        b.record_span("op", Duration::from_millis(20));
        b.record_span("other", Duration::from_millis(5));
        a.merge(&b);
        assert_eq!(a.get_region("op").unwrap().call_count, 2);
        assert_eq!(a.region_count(), 2);
    }

    #[test]
    fn test_total_time() {
        let mut p = Profiler::new();
        p.record_span("a", Duration::from_millis(10));
        p.record_span("b", Duration::from_millis(20));
        assert_eq!(p.total_time(), Duration::from_millis(30));
    }

    #[test]
    fn test_span_count() {
        let mut p = Profiler::new();
        p.record_span("a", Duration::from_millis(1));
        p.record_span("a", Duration::from_millis(1));
        p.record_span("b", Duration::from_millis(1));
        assert_eq!(p.span_count(), 3);
        assert_eq!(p.region_count(), 2);
    }

    #[test]
    fn test_start_span() {
        let mut p = Profiler::new();
        let guard = p.start_span("test");
        let _d = guard.finish();
        assert_eq!(p.get_region("test").unwrap().call_count, 1);
    }

    #[test]
    fn test_empty_region_avg() {
        let stats = RegionStats::new("empty");
        assert_eq!(stats.avg_time(), Duration::ZERO);
    }
}
