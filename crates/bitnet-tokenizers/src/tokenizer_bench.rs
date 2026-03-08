//! Tokenizer benchmarking utilities.
//!
//! Measure encode/decode throughput, latency, and efficiency
//! across different text lengths and patterns.

use std::time::{Duration, Instant};

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchRun {
    pub name: String,
    pub input_chars: usize,
    pub output_tokens: usize,
    pub encode_time: Duration,
    pub decode_time: Duration,
    pub iterations: usize,
}

impl BenchRun {
    pub fn chars_per_token(&self) -> f64 {
        if self.output_tokens == 0 {
            return 0.0;
        }
        self.input_chars as f64 / self.output_tokens as f64
    }

    pub fn encode_throughput(&self) -> f64 {
        let secs = self.encode_time.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (self.input_chars * self.iterations) as f64 / secs
    }

    pub fn decode_throughput(&self) -> f64 {
        let secs = self.decode_time.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        (self.output_tokens * self.iterations) as f64 / secs
    }

    pub fn avg_encode_latency(&self) -> Duration {
        if self.iterations == 0 {
            return Duration::ZERO;
        }
        self.encode_time / self.iterations as u32
    }

    pub fn avg_decode_latency(&self) -> Duration {
        if self.iterations == 0 {
            return Duration::ZERO;
        }
        self.decode_time / self.iterations as u32
    }
}

/// Benchmark suite results.
#[derive(Debug)]
pub struct BenchSuite {
    pub runs: Vec<BenchRun>,
}

impl BenchSuite {
    pub fn new() -> Self {
        Self { runs: Vec::new() }
    }

    pub fn add(&mut self, run: BenchRun) {
        self.runs.push(run);
    }

    pub fn total_encode_time(&self) -> Duration {
        self.runs.iter().map(|r| r.encode_time).sum()
    }

    pub fn total_decode_time(&self) -> Duration {
        self.runs.iter().map(|r| r.decode_time).sum()
    }

    pub fn avg_chars_per_token(&self) -> f64 {
        if self.runs.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.runs.iter().map(|r| r.chars_per_token()).sum();
        sum / self.runs.len() as f64
    }

    pub fn fastest_encode(&self) -> Option<&BenchRun> {
        self.runs.iter().min_by(|a, b| a.encode_time.cmp(&b.encode_time))
    }

    pub fn slowest_encode(&self) -> Option<&BenchRun> {
        self.runs.iter().max_by(|a, b| a.encode_time.cmp(&b.encode_time))
    }
}

impl Default for BenchSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate test texts of various lengths.
pub fn generate_test_texts() -> Vec<(&'static str, &'static str)> {
    vec![
        ("short", "Hello, world!"),
        (
            "medium",
            "The quick brown fox jumps over the lazy dog. This is a medium-length test sentence for tokenizer benchmarking.",
        ),
        ("code", "fn main() { let x: Vec<u32> = vec![1, 2, 3]; println!(\"{:?}\", x); }"),
        ("numbers", "3.14159 2.71828 1.41421 1.73205 2.23607 2.44949 2.64575 2.82843"),
        ("repeated", "token token token token token token token token token token"),
        (
            "special",
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>",
        ),
    ]
}

/// Measure encode performance with a simple char-splitting encoder (for testing).
pub fn bench_simple_encode(text: &str, iterations: usize) -> (Duration, usize) {
    let start = Instant::now();
    let mut token_count = 0;
    for _ in 0..iterations {
        // Simple whitespace tokenization as baseline
        token_count = text.split_whitespace().count();
    }
    (start.elapsed(), token_count)
}

/// Measure decode performance with simple token joining.
pub fn bench_simple_decode(tokens: &[&str], iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        let _: String = tokens.join(" ");
    }
    start.elapsed()
}

/// Run a complete benchmark on a text.
pub fn run_text_bench(name: &str, text: &str, iterations: usize) -> BenchRun {
    let (encode_time, token_count) = bench_simple_encode(text, iterations);
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let decode_time = bench_simple_decode(&tokens, iterations);

    BenchRun {
        name: name.to_string(),
        input_chars: text.len(),
        output_tokens: token_count,
        encode_time,
        decode_time,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_run_basic() {
        let run = run_text_bench("test", "hello world foo", 10);
        assert_eq!(run.output_tokens, 3);
        assert_eq!(run.iterations, 10);
    }

    #[test]
    fn test_chars_per_token() {
        let run = BenchRun {
            name: "t".into(),
            input_chars: 100,
            output_tokens: 25,
            encode_time: Duration::from_millis(1),
            decode_time: Duration::from_millis(1),
            iterations: 1,
        };
        assert!((run.chars_per_token() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_chars_per_token_zero() {
        let run = BenchRun {
            name: "t".into(),
            input_chars: 0,
            output_tokens: 0,
            encode_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            iterations: 1,
        };
        assert_eq!(run.chars_per_token(), 0.0);
    }

    #[test]
    fn test_encode_throughput() {
        let run = run_text_bench("test", "hello world", 100);
        assert!(run.encode_throughput() > 0.0);
    }

    #[test]
    fn test_avg_latency() {
        let run = run_text_bench("test", "hello world", 10);
        assert!(run.avg_encode_latency() < Duration::from_secs(1));
    }

    #[test]
    fn test_suite() {
        let mut suite = BenchSuite::new();
        suite.add(run_text_bench("a", "hello", 5));
        suite.add(run_text_bench("b", "hello world foo", 5));
        assert_eq!(suite.runs.len(), 2);
        assert!(suite.avg_chars_per_token() > 0.0);
    }

    #[test]
    fn test_fastest_slowest() {
        let mut suite = BenchSuite::new();
        suite.add(run_text_bench("short", "hi", 10));
        suite.add(run_text_bench("long", "a b c d e f g h i j k l m n", 10));
        assert!(suite.fastest_encode().is_some());
        assert!(suite.slowest_encode().is_some());
    }

    #[test]
    fn test_generate_texts() {
        let texts = generate_test_texts();
        assert!(texts.len() >= 5);
    }

    #[test]
    fn test_simple_encode() {
        let (dur, count) = bench_simple_encode("hello world", 1);
        assert_eq!(count, 2);
        assert!(dur < Duration::from_secs(1));
    }

    #[test]
    fn test_simple_decode() {
        let dur = bench_simple_decode(&["hello", "world"], 1);
        assert!(dur < Duration::from_secs(1));
    }

    #[test]
    fn test_total_times() {
        let mut suite = BenchSuite::new();
        suite.add(run_text_bench("a", "hello", 5));
        assert!(suite.total_encode_time() >= Duration::ZERO);
    }

    #[test]
    fn test_empty_suite() {
        let suite = BenchSuite::new();
        assert_eq!(suite.avg_chars_per_token(), 0.0);
        assert!(suite.fastest_encode().is_none());
    }
}
