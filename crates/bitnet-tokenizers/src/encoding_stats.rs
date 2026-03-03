//! Tokenizer encoding statistics.
//!
//! Track encoding lengths, token distributions, and OOV rates.

use std::collections::HashMap;

/// Statistics from encoding text.
#[derive(Debug, Clone)]
pub struct EncodingStats {
    pub total_texts: usize,
    pub total_tokens: u64,
    pub total_chars: u64,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub oov_count: u64,
    pub token_frequencies: HashMap<u32, u64>,
    length_distribution: Vec<usize>,
}

impl Default for EncodingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingStats {
    pub fn new() -> Self {
        Self {
            total_texts: 0,
            total_tokens: 0,
            total_chars: 0,
            min_tokens: usize::MAX,
            max_tokens: 0,
            oov_count: 0,
            token_frequencies: HashMap::new(),
            length_distribution: Vec::new(),
        }
    }

    /// Record an encoded sequence.
    pub fn record(&mut self, text: &str, token_ids: &[u32], oov_token_id: Option<u32>) {
        self.total_texts += 1;
        let len = token_ids.len();
        self.total_tokens += len as u64;
        self.total_chars += text.len() as u64;
        if len < self.min_tokens {
            self.min_tokens = len;
        }
        if len > self.max_tokens {
            self.max_tokens = len;
        }

        for &id in token_ids {
            *self.token_frequencies.entry(id).or_insert(0) += 1;
            if oov_token_id == Some(id) {
                self.oov_count += 1;
            }
        }

        if len >= self.length_distribution.len() {
            self.length_distribution.resize(len + 1, 0);
        }
        self.length_distribution[len] += 1;
    }

    /// Average tokens per text.
    pub fn avg_tokens(&self) -> f64 {
        if self.total_texts == 0 {
            return 0.0;
        }
        self.total_tokens as f64 / self.total_texts as f64
    }

    /// Compression ratio (chars per token).
    pub fn compression_ratio(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        self.total_chars as f64 / self.total_tokens as f64
    }

    /// OOV rate as fraction.
    pub fn oov_rate(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        self.oov_count as f64 / self.total_tokens as f64
    }

    /// Number of unique tokens seen.
    pub fn unique_tokens(&self) -> usize {
        self.token_frequencies.len()
    }

    /// Top N most frequent tokens.
    pub fn top_tokens(&self, n: usize) -> Vec<(u32, u64)> {
        let mut pairs: Vec<_> = self.token_frequencies.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(n);
        pairs
    }

    /// Length distribution as (length, count) pairs.
    pub fn length_histogram(&self) -> Vec<(usize, usize)> {
        self.length_distribution
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(len, &count)| (len, count))
            .collect()
    }

    /// Merge another stats into this one.
    pub fn merge(&mut self, other: &EncodingStats) {
        self.total_texts += other.total_texts;
        self.total_tokens += other.total_tokens;
        self.total_chars += other.total_chars;
        self.min_tokens = self.min_tokens.min(other.min_tokens);
        self.max_tokens = self.max_tokens.max(other.max_tokens);
        self.oov_count += other.oov_count;
        for (&id, &count) in &other.token_frequencies {
            *self.token_frequencies.entry(id).or_insert(0) += count;
        }
        if other.length_distribution.len() > self.length_distribution.len() {
            self.length_distribution.resize(other.length_distribution.len(), 0);
        }
        for (i, &c) in other.length_distribution.iter().enumerate() {
            self.length_distribution[i] += c;
        }
    }
}

/// Summary for display.
#[derive(Debug, Clone)]
pub struct StatsSummary {
    pub total_texts: usize,
    pub total_tokens: u64,
    pub avg_tokens: f64,
    pub compression_ratio: f64,
    pub oov_rate: f64,
    pub unique_tokens: usize,
    pub min_len: usize,
    pub max_len: usize,
}

impl EncodingStats {
    pub fn summary(&self) -> StatsSummary {
        StatsSummary {
            total_texts: self.total_texts,
            total_tokens: self.total_tokens,
            avg_tokens: self.avg_tokens(),
            compression_ratio: self.compression_ratio(),
            oov_rate: self.oov_rate(),
            unique_tokens: self.unique_tokens(),
            min_len: if self.min_tokens == usize::MAX { 0 } else { self.min_tokens },
            max_len: self.max_tokens,
        }
    }
}

/// Batch analyzer.
#[derive(Debug)]
pub struct BatchAnalyzer {
    oov_id: Option<u32>,
    stats: EncodingStats,
}

impl BatchAnalyzer {
    pub fn new(oov_id: Option<u32>) -> Self {
        Self { oov_id, stats: EncodingStats::new() }
    }

    pub fn add(&mut self, text: &str, token_ids: &[u32]) {
        self.stats.record(text, token_ids, self.oov_id);
    }

    pub fn finish(self) -> EncodingStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_stats() {
        let stats = EncodingStats::new();
        assert_eq!(stats.total_texts, 0);
        assert_eq!(stats.avg_tokens(), 0.0);
        assert_eq!(stats.oov_rate(), 0.0);
    }

    #[test]
    fn test_single_record() {
        let mut stats = EncodingStats::new();
        stats.record("hello world", &[1, 2], None);
        assert_eq!(stats.total_texts, 1);
        assert_eq!(stats.total_tokens, 2);
        assert_eq!(stats.avg_tokens(), 2.0);
    }

    #[test]
    fn test_compression_ratio() {
        let mut stats = EncodingStats::new();
        stats.record("hello", &[1, 2], None);
        assert!((stats.compression_ratio() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_oov_tracking() {
        let mut stats = EncodingStats::new();
        stats.record("test", &[1, 0, 2, 0], Some(0));
        assert_eq!(stats.oov_count, 2);
        assert!((stats.oov_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_min_max_tokens() {
        let mut stats = EncodingStats::new();
        stats.record("a", &[1], None);
        stats.record("b c d", &[2, 3, 4], None);
        assert_eq!(stats.min_tokens, 1);
        assert_eq!(stats.max_tokens, 3);
    }

    #[test]
    fn test_unique_tokens() {
        let mut stats = EncodingStats::new();
        stats.record("a", &[1, 2, 1], None);
        assert_eq!(stats.unique_tokens(), 2);
    }

    #[test]
    fn test_top_tokens() {
        let mut stats = EncodingStats::new();
        stats.record("a", &[1, 1, 1, 2, 2, 3], None);
        let top = stats.top_tokens(2);
        assert_eq!(top[0], (1, 3));
        assert_eq!(top[1], (2, 2));
    }

    #[test]
    fn test_length_histogram() {
        let mut stats = EncodingStats::new();
        stats.record("a", &[1], None);
        stats.record("b", &[2], None);
        stats.record("c", &[3, 4, 5], None);
        let hist = stats.length_histogram();
        assert!(hist.contains(&(1, 2)));
        assert!(hist.contains(&(3, 1)));
    }

    #[test]
    fn test_merge() {
        let mut a = EncodingStats::new();
        a.record("x", &[1, 2], None);
        let mut b = EncodingStats::new();
        b.record("y", &[3], None);
        a.merge(&b);
        assert_eq!(a.total_texts, 2);
        assert_eq!(a.total_tokens, 3);
    }

    #[test]
    fn test_summary() {
        let mut stats = EncodingStats::new();
        stats.record("hello", &[1, 2], None);
        let s = stats.summary();
        assert_eq!(s.total_texts, 1);
        assert_eq!(s.total_tokens, 2);
    }

    #[test]
    fn test_batch_analyzer() {
        let mut analyzer = BatchAnalyzer::new(Some(99));
        analyzer.add("test", &[1, 99, 2]);
        let stats = analyzer.finish();
        assert_eq!(stats.oov_count, 1);
    }

    #[test]
    fn test_empty_summary_min_len() {
        let stats = EncodingStats::new();
        assert_eq!(stats.summary().min_len, 0);
    }
}
