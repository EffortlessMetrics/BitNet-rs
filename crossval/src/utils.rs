//! Utility functions for cross-validation

use crate::{CrossvalConfig, CrossvalError, Result};

/// Compare two sequences of tokens for numerical equivalence
pub fn compare_tokens(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    config: &CrossvalConfig,
) -> Result<bool> {
    if rust_tokens.len() != cpp_tokens.len() {
        return Err(CrossvalError::ComparisonError(format!(
            "Token sequence length mismatch: Rust={}, C++={}",
            rust_tokens.len(),
            cpp_tokens.len()
        )));
    }

    let max_compare = config.max_tokens.min(rust_tokens.len());

    for i in 0..max_compare {
        if rust_tokens[i] != cpp_tokens[i] {
            return Err(CrossvalError::ComparisonError(format!(
                "Token mismatch at position {}: Rust={}, C++={}",
                i, rust_tokens[i], cpp_tokens[i]
            )));
        }
    }

    Ok(true)
}

/// Compare two sequences of floating-point values with tolerance
pub fn compare_floats(
    rust_values: &[f32],
    cpp_values: &[f32],
    config: &CrossvalConfig,
) -> Result<bool> {
    if rust_values.len() != cpp_values.len() {
        return Err(CrossvalError::ComparisonError(format!(
            "Float sequence length mismatch: Rust={}, C++={}",
            rust_values.len(),
            cpp_values.len()
        )));
    }

    for (i, (&rust_val, &cpp_val)) in rust_values.iter().zip(cpp_values.iter()).enumerate() {
        let diff = (rust_val - cpp_val).abs();
        let tolerance = config.tolerance as f32;

        if diff > tolerance {
            return Err(CrossvalError::ComparisonError(format!(
                "Float value mismatch at position {}: Rust={}, C++={}, diff={} > tolerance={}",
                i, rust_val, cpp_val, diff, tolerance
            )));
        }
    }

    Ok(true)
}

/// Performance measurement utilities
pub mod perf {
    use std::time::{Duration, Instant};

    /// Simple performance measurement
    pub struct PerfMeasurement {
        pub duration: Duration,
        pub tokens_per_second: f64,
    }

    /// Measure performance of a closure that returns token count
    pub fn measure<F>(f: F) -> (PerfMeasurement, F::Output)
    where
        F: FnOnce() -> usize,
    {
        let start = Instant::now();
        let token_count = f();
        let duration = start.elapsed();

        let tokens_per_second = if duration.as_secs_f64() > 0.0 {
            token_count as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        (
            PerfMeasurement {
                duration,
                tokens_per_second,
            },
            token_count,
        )
    }
}

/// Logging utilities for cross-validation
pub mod logging {
    /// Log a comparison result
    pub fn log_comparison(test_name: &str, rust_tokens: usize, cpp_tokens: usize, success: bool) {
        let status = if success { "✓ PASS" } else { "✗ FAIL" };
        println!(
            "[{}] {}: Rust={} tokens, C++={} tokens",
            status, test_name, rust_tokens, cpp_tokens
        );
    }

    /// Log performance comparison
    pub fn log_performance(test_name: &str, rust_tps: f64, cpp_tps: f64) {
        let speedup = if cpp_tps > 0.0 {
            rust_tps / cpp_tps
        } else {
            0.0
        };
        println!(
            "[PERF] {}: Rust={:.1} tok/s, C++={:.1} tok/s, speedup={:.2}x",
            test_name, rust_tps, cpp_tps, speedup
        );
    }
}
