//! Tokenizer benchmarking utilities.
//!
//! Corpus generators and timing helpers for tokenizer performance testing.

use std::time::Instant;

/// A benchmark corpus with known characteristics.
#[derive(Debug, Clone)]
pub struct BenchCorpus {
    pub name: String,
    pub text: String,
    pub expected_min_tokens: usize,
}

/// Generate standard benchmark corpora.
pub fn standard_corpora() -> Vec<BenchCorpus> {
    vec![
        BenchCorpus {
            name: "english_prose".into(),
            text: "The quick brown fox jumps over the lazy dog. ".repeat(100),
            expected_min_tokens: 400,
        },
        BenchCorpus {
            name: "code_python".into(),
            text: "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n".repeat(50),
            expected_min_tokens: 500,
        },
        BenchCorpus {
            name: "code_rust".into(),
            text: "fn main() {\n    let x: Vec<u32> = (0..100).collect();\n    println!(\"{:?}\", x);\n}\n".repeat(50),
            expected_min_tokens: 400,
        },
        BenchCorpus {
            name: "numbers".into(),
            text: (0..500).map(|i| i.to_string()).collect::<Vec<_>>().join(" "),
            expected_min_tokens: 200,
        },
        BenchCorpus {
            name: "mixed_unicode".into(),
            text: "Hello 你好 مرحبا Привет こんにちは 🌍🎉 ".repeat(50),
            expected_min_tokens: 100,
        },
        BenchCorpus {
            name: "whitespace_heavy".into(),
            text: "word   \t\n  word   \t\n  ".repeat(200),
            expected_min_tokens: 100,
        },
        BenchCorpus {
            name: "single_char".into(),
            text: "a".repeat(1000),
            expected_min_tokens: 1,
        },
        BenchCorpus {
            name: "json_like".into(),
            text: r#"{"key": "value", "num": 42, "arr": [1,2,3]}"#.repeat(50).to_string(),
            expected_min_tokens: 200,
        },
    ]
}

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub corpus_name: String,
    pub input_bytes: usize,
    pub token_count: usize,
    pub encode_us: u64,
    pub decode_us: u64,
    pub bytes_per_token: f32,
    pub tokens_per_sec: f64,
}

impl BenchResult {
    pub fn compression_ratio(&self) -> f32 {
        if self.token_count == 0 {
            return 0.0;
        }
        self.input_bytes as f32 / self.token_count as f32
    }
}

/// Time an encoding operation (returns microseconds).
pub fn time_encode<F>(text: &str, mut encode_fn: F) -> (u64, Vec<u32>)
where
    F: FnMut(&str) -> Vec<u32>,
{
    let start = Instant::now();
    let tokens = encode_fn(text);
    let us = start.elapsed().as_micros() as u64;
    (us, tokens)
}

/// Time a decoding operation (returns microseconds).
pub fn time_decode<F>(tokens: &[u32], mut decode_fn: F) -> (u64, String)
where
    F: FnMut(&[u32]) -> String,
{
    let start = Instant::now();
    let text = decode_fn(tokens);
    let us = start.elapsed().as_micros() as u64;
    (us, text)
}

/// Run a complete benchmark with encode+decode, returning results.
pub fn run_benchmark<E, D>(
    corpus: &BenchCorpus,
    mut encode_fn: E,
    mut decode_fn: D,
) -> BenchResult
where
    E: FnMut(&str) -> Vec<u32>,
    D: FnMut(&[u32]) -> String,
{
    let (enc_us, tokens) = time_encode(&corpus.text, &mut encode_fn);
    let (dec_us, _decoded) = time_decode(&tokens, &mut decode_fn);

    let tps = if enc_us > 0 {
        tokens.len() as f64 / (enc_us as f64 / 1_000_000.0)
    } else {
        0.0
    };

    BenchResult {
        corpus_name: corpus.name.clone(),
        input_bytes: corpus.text.len(),
        token_count: tokens.len(),
        encode_us: enc_us,
        decode_us: dec_us,
        bytes_per_token: if tokens.is_empty() {
            0.0
        } else {
            corpus.text.len() as f32 / tokens.len() as f32
        },
        tokens_per_sec: tps,
    }
}

/// Aggregate multiple benchmark results.
#[derive(Debug, Clone)]
pub struct BenchSummary {
    pub total_bytes: usize,
    pub total_tokens: usize,
    pub total_encode_us: u64,
    pub total_decode_us: u64,
    pub avg_bytes_per_token: f32,
    pub avg_tokens_per_sec: f64,
}

pub fn summarize(results: &[BenchResult]) -> BenchSummary {
    if results.is_empty() {
        return BenchSummary {
            total_bytes: 0,
            total_tokens: 0,
            total_encode_us: 0,
            total_decode_us: 0,
            avg_bytes_per_token: 0.0,
            avg_tokens_per_sec: 0.0,
        };
    }

    let total_bytes: usize = results.iter().map(|r| r.input_bytes).sum();
    let total_tokens: usize = results.iter().map(|r| r.token_count).sum();
    let total_encode_us: u64 = results.iter().map(|r| r.encode_us).sum();
    let total_decode_us: u64 = results.iter().map(|r| r.decode_us).sum();

    let avg_bpt = if total_tokens > 0 {
        total_bytes as f32 / total_tokens as f32
    } else {
        0.0
    };
    let avg_tps = if total_encode_us > 0 {
        total_tokens as f64 / (total_encode_us as f64 / 1_000_000.0)
    } else {
        0.0
    };

    BenchSummary {
        total_bytes,
        total_tokens,
        total_encode_us,
        total_decode_us,
        avg_bytes_per_token: avg_bpt,
        avg_tokens_per_sec: avg_tps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_corpora_count() {
        let c = standard_corpora();
        assert_eq!(c.len(), 8);
    }

    #[test]
    fn test_corpus_non_empty() {
        for c in standard_corpora() {
            assert!(!c.text.is_empty(), "{} is empty", c.name);
        }
    }

    #[test]
    fn test_time_encode() {
        let (us, tokens) = time_encode("hello", |s| {
            s.bytes().map(|b| b as u32).collect()
        });
        assert_eq!(tokens.len(), 5);
        assert!(us < 1_000_000); // < 1 second
    }

    #[test]
    fn test_time_decode() {
        let (us, text) = time_decode(&[104, 105], |t| {
            t.iter().map(|&b| b as u8 as char).collect()
        });
        assert_eq!(text, "hi");
        assert!(us < 1_000_000);
    }

    #[test]
    fn test_run_benchmark() {
        let corpus = BenchCorpus {
            name: "test".into(),
            text: "hello world".into(),
            expected_min_tokens: 2,
        };
        let result = run_benchmark(
            &corpus,
            |s| {
                s.split_whitespace()
                    .enumerate()
                    .map(|(i, _)| i as u32)
                    .collect()
            },
            |_t| "hello world".into(),
        );
        assert_eq!(result.token_count, 2);
        assert_eq!(result.input_bytes, 11);
    }

    #[test]
    fn test_compression_ratio() {
        let r = BenchResult {
            corpus_name: "t".into(),
            input_bytes: 100,
            token_count: 25,
            encode_us: 10,
            decode_us: 5,
            bytes_per_token: 4.0,
            tokens_per_sec: 1000.0,
        };
        assert!((r.compression_ratio() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_compression_ratio_zero() {
        let r = BenchResult {
            corpus_name: "t".into(),
            input_bytes: 100,
            token_count: 0,
            encode_us: 0,
            decode_us: 0,
            bytes_per_token: 0.0,
            tokens_per_sec: 0.0,
        };
        assert_eq!(r.compression_ratio(), 0.0);
    }

    #[test]
    fn test_summarize() {
        let results = vec![
            BenchResult {
                corpus_name: "a".into(),
                input_bytes: 100,
                token_count: 25,
                encode_us: 1000,
                decode_us: 500,
                bytes_per_token: 4.0,
                tokens_per_sec: 25000.0,
            },
            BenchResult {
                corpus_name: "b".into(),
                input_bytes: 200,
                token_count: 50,
                encode_us: 2000,
                decode_us: 1000,
                bytes_per_token: 4.0,
                tokens_per_sec: 25000.0,
            },
        ];
        let s = summarize(&results);
        assert_eq!(s.total_bytes, 300);
        assert_eq!(s.total_tokens, 75);
    }

    #[test]
    fn test_summarize_empty() {
        let s = summarize(&[]);
        assert_eq!(s.total_tokens, 0);
    }

    #[test]
    fn test_corpus_names_unique() {
        let c = standard_corpora();
        let names: Vec<&str> = c.iter().map(|x| x.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len());
    }

    #[test]
    fn test_bench_result_fields() {
        let corpus = BenchCorpus {
            name: "x".into(),
            text: "abc".into(),
            expected_min_tokens: 1,
        };
        let r = run_benchmark(&corpus, |_| vec![1, 2, 3], |_| "abc".into());
        assert_eq!(r.corpus_name, "x");
        assert_eq!(r.token_count, 3);
        assert!(r.bytes_per_token > 0.0);
    }

    #[test]
    fn test_english_corpus_size() {
        let c = standard_corpora();
        let eng = c.iter().find(|x| x.name == "english_prose").unwrap();
        assert!(eng.text.len() > 1000);
    }
}
