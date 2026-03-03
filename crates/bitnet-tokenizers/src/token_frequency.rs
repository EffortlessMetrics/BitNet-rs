//! Token frequency analysis for encoded text.
//!
//! Count token frequencies, compute coverage metrics,
//! identify most/least common tokens, and entropy.

use std::collections::HashMap;

/// Token frequency counter.
#[derive(Debug, Default)]
pub struct FrequencyCounter {
    counts: HashMap<u32, usize>,
    total: usize,
}

impl FrequencyCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single token occurrence.
    pub fn record(&mut self, token_id: u32) {
        *self.counts.entry(token_id).or_insert(0) += 1;
        self.total += 1;
    }

    /// Record a batch of token IDs.
    pub fn record_batch(&mut self, token_ids: &[u32]) {
        for &id in token_ids {
            self.record(id);
        }
    }

    /// Get count for a specific token.
    pub fn count(&self, token_id: u32) -> usize {
        self.counts.get(&token_id).copied().unwrap_or(0)
    }

    /// Total number of tokens recorded.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of unique tokens seen.
    pub fn unique_count(&self) -> usize {
        self.counts.len()
    }

    /// Relative frequency of a token (0.0 to 1.0).
    pub fn frequency(&self, token_id: u32) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count(token_id) as f64 / self.total as f64
    }

    /// Top-k most frequent tokens.
    pub fn top_k(&self, k: usize) -> Vec<(u32, usize)> {
        let mut entries: Vec<(u32, usize)> = self.counts.iter().map(|(&id, &c)| (id, c)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(k);
        entries
    }

    /// Bottom-k least frequent tokens.
    pub fn bottom_k(&self, k: usize) -> Vec<(u32, usize)> {
        let mut entries: Vec<(u32, usize)> = self.counts.iter().map(|(&id, &c)| (id, c)).collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1));
        entries.truncate(k);
        entries
    }

    /// Shannon entropy of the distribution (bits).
    pub fn entropy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mut h = 0.0f64;
        for &count in self.counts.values() {
            if count > 0 {
                let p = count as f64 / self.total as f64;
                h -= p * p.log2();
            }
        }
        h
    }

    /// Merge another counter into this one.
    pub fn merge(&mut self, other: &FrequencyCounter) {
        for (&id, &count) in &other.counts {
            *self.counts.entry(id).or_insert(0) += count;
        }
        self.total += other.total;
    }

    /// Coverage: what fraction of total tokens do the top-k unique tokens cover?
    pub fn coverage_top_k(&self, k: usize) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let top = self.top_k(k);
        let covered: usize = top.iter().map(|(_, c)| c).sum();
        covered as f64 / self.total as f64
    }

    /// Clear all counts.
    pub fn clear(&mut self) {
        self.counts.clear();
        self.total = 0;
    }
}

/// Frequency report summary.
#[derive(Debug)]
pub struct FrequencyReport {
    pub total_tokens: usize,
    pub unique_tokens: usize,
    pub entropy_bits: f64,
    pub top_5: Vec<(u32, usize)>,
    pub coverage_top_100: f64,
}

/// Generate a frequency report from a counter.
pub fn generate_report(counter: &FrequencyCounter) -> FrequencyReport {
    FrequencyReport {
        total_tokens: counter.total(),
        unique_tokens: counter.unique_count(),
        entropy_bits: counter.entropy(),
        top_5: counter.top_k(5),
        coverage_top_100: counter.coverage_top_k(100),
    }
}

impl FrequencyReport {
    pub fn summary(&self) -> String {
        format!(
            "tokens={}, unique={}, entropy={:.2}bits, top100_coverage={:.1}%",
            self.total_tokens,
            self.unique_tokens,
            self.entropy_bits,
            self.coverage_top_100 * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_count() {
        let mut counter = FrequencyCounter::new();
        counter.record(42);
        counter.record(42);
        counter.record(7);
        assert_eq!(counter.count(42), 2);
        assert_eq!(counter.count(7), 1);
        assert_eq!(counter.count(99), 0);
    }

    #[test]
    fn test_batch_record() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 2, 3, 1, 2, 1]);
        assert_eq!(counter.total(), 6);
        assert_eq!(counter.unique_count(), 3);
        assert_eq!(counter.count(1), 3);
    }

    #[test]
    fn test_frequency() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 1, 2, 3]);
        assert!((counter.frequency(1) - 0.5).abs() < 1e-6);
        assert!((counter.frequency(2) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_top_k() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 1, 1, 2, 2, 3]);
        let top = counter.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[0].1, 3);
    }

    #[test]
    fn test_bottom_k() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 1, 1, 2, 2, 3]);
        let bottom = counter.bottom_k(1);
        assert_eq!(bottom[0].0, 3);
        assert_eq!(bottom[0].1, 1);
    }

    #[test]
    fn test_entropy() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 2]); // uniform: log2(2)=1 bit
        assert!((counter.entropy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_entropy_empty() {
        let counter = FrequencyCounter::new();
        assert_eq!(counter.entropy(), 0.0);
    }

    #[test]
    fn test_merge() {
        let mut c1 = FrequencyCounter::new();
        c1.record_batch(&[1, 2]);
        let mut c2 = FrequencyCounter::new();
        c2.record_batch(&[2, 3]);
        c1.merge(&c2);
        assert_eq!(c1.total(), 4);
        assert_eq!(c1.count(2), 2);
    }

    #[test]
    fn test_coverage() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 1, 1, 2]);
        assert!((counter.coverage_top_k(1) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_clear() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 2, 3]);
        counter.clear();
        assert_eq!(counter.total(), 0);
        assert_eq!(counter.unique_count(), 0);
    }

    #[test]
    fn test_report() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 2, 3, 1, 2, 1]);
        let report = generate_report(&counter);
        assert_eq!(report.total_tokens, 6);
        assert_eq!(report.unique_tokens, 3);
    }

    #[test]
    fn test_report_summary() {
        let mut counter = FrequencyCounter::new();
        counter.record_batch(&[1, 2]);
        let report = generate_report(&counter);
        let s = report.summary();
        assert!(s.contains("tokens=2"));
        assert!(s.contains("unique=2"));
    }
}
