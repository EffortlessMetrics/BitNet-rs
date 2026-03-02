//! Model profiler.
//!
//! Layer-by-layer timing and performance analysis for inference.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Layer type for profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    Embedding,
    Attention,
    FeedForward,
    Normalization,
    Projection,
    Activation,
    Other,
}

impl LayerType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Attention => "attention",
            Self::FeedForward => "feed_forward",
            Self::Normalization => "normalization",
            Self::Projection => "projection",
            Self::Activation => "activation",
            Self::Other => "other",
        }
    }
}

/// Timing record for a single operation.
#[derive(Debug, Clone)]
pub struct TimingRecord {
    pub layer_type: LayerType,
    pub layer_index: usize,
    pub duration: Duration,
    pub label: String,
}

/// Active timer handle.
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    layer_type: LayerType,
    layer_index: usize,
    label: String,
}

/// Profiler collecting timing data.
#[derive(Debug)]
pub struct ModelProfiler {
    records: Vec<TimingRecord>,
    active: Option<Timer>,
    enabled: bool,
}

impl Default for ModelProfiler {
    fn default() -> Self {
        Self::new(true)
    }
}

impl ModelProfiler {
    pub fn new(enabled: bool) -> Self {
        Self { records: Vec::new(), active: None, enabled }
    }

    pub fn start(&mut self, layer_type: LayerType, layer_index: usize, label: &str) {
        if !self.enabled {
            return;
        }
        self.active = Some(Timer {
            start: Instant::now(),
            layer_type,
            layer_index,
            label: label.to_string(),
        });
    }

    pub fn stop(&mut self) {
        if let Some(timer) = self.active.take() {
            self.records.push(TimingRecord {
                layer_type: timer.layer_type,
                layer_index: timer.layer_index,
                duration: timer.start.elapsed(),
                label: timer.label,
            });
        }
    }

    pub fn record(
        &mut self,
        layer_type: LayerType,
        layer_index: usize,
        label: &str,
        duration: Duration,
    ) {
        if !self.enabled {
            return;
        }
        self.records.push(TimingRecord {
            layer_type,
            layer_index,
            duration,
            label: label.to_string(),
        });
    }

    pub fn records(&self) -> &[TimingRecord] {
        &self.records
    }

    pub fn count(&self) -> usize {
        self.records.len()
    }

    pub fn total_time(&self) -> Duration {
        self.records.iter().map(|r| r.duration).sum()
    }

    pub fn time_by_type(&self) -> HashMap<LayerType, Duration> {
        let mut map = HashMap::new();
        for r in &self.records {
            *map.entry(r.layer_type).or_insert(Duration::ZERO) += r.duration;
        }
        map
    }

    pub fn slowest(&self, n: usize) -> Vec<&TimingRecord> {
        let mut sorted: Vec<&TimingRecord> = self.records.iter().collect();
        sorted.sort_by(|a, b| b.duration.cmp(&a.duration));
        sorted.truncate(n);
        sorted
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Generate a profile summary.
    pub fn summary(&self) -> ProfileSummary {
        let total = self.total_time();
        let by_type = self.time_by_type();
        let mut breakdown: Vec<(LayerType, Duration, f64)> = by_type
            .into_iter()
            .map(|(t, d)| {
                let pct = if total.as_nanos() > 0 {
                    d.as_nanos() as f64 / total.as_nanos() as f64 * 100.0
                } else {
                    0.0
                };
                (t, d, pct)
            })
            .collect();
        breakdown.sort_by(|a, b| b.1.cmp(&a.1));

        ProfileSummary { total, record_count: self.records.len(), breakdown }
    }
}

/// Profile summary.
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub total: Duration,
    pub record_count: usize,
    /// Type, time, percentage.
    pub breakdown: Vec<(LayerType, Duration, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_profiler() {
        let p = ModelProfiler::new(true);
        assert_eq!(p.count(), 0);
    }

    #[test]
    fn test_record() {
        let mut p = ModelProfiler::new(true);
        p.record(LayerType::Attention, 0, "attn0", Duration::from_millis(10));
        assert_eq!(p.count(), 1);
    }

    #[test]
    fn test_total_time() {
        let mut p = ModelProfiler::new(true);
        p.record(LayerType::Attention, 0, "a", Duration::from_millis(10));
        p.record(LayerType::FeedForward, 0, "f", Duration::from_millis(20));
        assert_eq!(p.total_time(), Duration::from_millis(30));
    }

    #[test]
    fn test_time_by_type() {
        let mut p = ModelProfiler::new(true);
        p.record(LayerType::Attention, 0, "a0", Duration::from_millis(10));
        p.record(LayerType::Attention, 1, "a1", Duration::from_millis(15));
        p.record(LayerType::FeedForward, 0, "f0", Duration::from_millis(5));
        let by_type = p.time_by_type();
        assert_eq!(by_type[&LayerType::Attention], Duration::from_millis(25));
        assert_eq!(by_type[&LayerType::FeedForward], Duration::from_millis(5));
    }

    #[test]
    fn test_slowest() {
        let mut p = ModelProfiler::new(true);
        p.record(LayerType::Attention, 0, "slow", Duration::from_millis(100));
        p.record(LayerType::FeedForward, 0, "fast", Duration::from_millis(1));
        p.record(LayerType::Normalization, 0, "mid", Duration::from_millis(50));
        let top = p.slowest(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].label, "slow");
    }

    #[test]
    fn test_disabled() {
        let mut p = ModelProfiler::new(false);
        p.record(LayerType::Attention, 0, "a", Duration::from_millis(10));
        assert_eq!(p.count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut p = ModelProfiler::new(true);
        p.record(LayerType::Attention, 0, "a", Duration::from_millis(10));
        p.clear();
        assert_eq!(p.count(), 0);
    }

    #[test]
    fn test_start_stop() {
        let mut p = ModelProfiler::new(true);
        p.start(LayerType::Embedding, 0, "embed");
        std::thread::sleep(Duration::from_millis(1));
        p.stop();
        assert_eq!(p.count(), 1);
        assert!(p.records()[0].duration >= Duration::from_millis(1));
    }

    #[test]
    fn test_summary() {
        let mut p = ModelProfiler::new(true);
        p.record(LayerType::Attention, 0, "a", Duration::from_millis(70));
        p.record(LayerType::FeedForward, 0, "f", Duration::from_millis(30));
        let s = p.summary();
        assert_eq!(s.record_count, 2);
        assert_eq!(s.total, Duration::from_millis(100));
        assert!(!s.breakdown.is_empty());
    }

    #[test]
    fn test_summary_percentages() {
        let mut p = ModelProfiler::new(true);
        p.record(LayerType::Attention, 0, "a", Duration::from_millis(75));
        p.record(LayerType::FeedForward, 0, "f", Duration::from_millis(25));
        let s = p.summary();
        let attn_pct = s.breakdown.iter().find(|(t, _, _)| *t == LayerType::Attention).unwrap().2;
        assert!((attn_pct - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_layer_type_str() {
        assert_eq!(LayerType::Attention.as_str(), "attention");
        assert_eq!(LayerType::FeedForward.as_str(), "feed_forward");
    }

    #[test]
    fn test_default() {
        let p = ModelProfiler::default();
        assert_eq!(p.count(), 0);
    }
}
