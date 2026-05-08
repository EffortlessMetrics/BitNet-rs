//! Rust benchmark companion for the Python-to-Rust migration example.
//!
//! This is a dependency-free smoke benchmark intended for documentation and
//! migration validation, not a statistically rigorous Criterion benchmark.

use std::error::Error;
use std::time::{Duration, Instant};

#[path = "main.rs"]
mod inference;

use inference::{BitNetInference, GenerationConfig};

const PROMPTS: &[&str] = &[
    "The future of AI is",
    "Rust programming language",
    "Machine learning models",
    "High performance computing",
];

#[derive(Debug, Clone, PartialEq)]
struct BenchmarkSummary {
    avg_time: Duration,
    tokens_per_second: f64,
    total_tokens: usize,
}

fn benchmark_inference(
    model_path: &str,
    prompts: &[&str],
    runs: usize,
) -> Result<BenchmarkSummary, Box<dyn Error>> {
    let model = BitNetInference::new(model_path)?;
    let mut total_time = Duration::ZERO;
    let mut total_tokens = 0;
    let mut samples = 0;

    for _ in 0..runs {
        for prompt in prompts {
            let started_at = Instant::now();
            let result = model.generate(prompt, GenerationConfig { max_tokens: 50 })?;
            total_time += started_at.elapsed();
            total_tokens += result.tokens;
            samples += 1;
        }
    }

    let avg_time = if samples == 0 {
        Duration::ZERO
    } else {
        Duration::from_secs_f64(total_time.as_secs_f64() / samples as f64)
    };
    let tokens_per_second =
        if total_time.is_zero() { 0.0 } else { total_tokens as f64 / total_time.as_secs_f64() };

    Ok(BenchmarkSummary { avg_time, tokens_per_second, total_tokens })
}

fn main() -> Result<(), Box<dyn Error>> {
    let results = benchmark_inference("model.gguf", PROMPTS, 5)?;

    println!("Rust implementation: {:.1} tok/s", results.tokens_per_second);
    println!("Average request time: {:.3} ms", results.avg_time.as_secs_f64() * 1000.0);
    println!("Total tokens: {}", results.total_tokens);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_produces_summary() {
        let results = benchmark_inference("model.gguf", PROMPTS, 1).unwrap();

        assert_eq!(results.total_tokens, PROMPTS.len() * 50);
        assert!(results.tokens_per_second >= 0.0);
    }
}
