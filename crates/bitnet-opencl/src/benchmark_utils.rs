//! Cross-backend benchmark utilities: result collection, statistics,
//! comparison tables, and human-readable report formatting.

use serde::{Deserialize, Serialize};
use std::fmt::{self, Write as _};
use std::time::Duration;

// ── BenchmarkResult ─────────────────────────────────────────────────────────

/// Collected timing and throughput data from a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Human-readable name of the benchmark.
    pub name: String,
    /// Backend that produced these results (e.g. "cpu", "opencl", "vulkan").
    pub backend: String,
    /// Individual sample durations collected over N iterations.
    pub samples: Vec<Duration>,
    /// Total number of logical elements processed per iteration.
    pub elements_per_iter: u64,
    /// Peak memory usage in bytes (optional, backend-dependent).
    pub peak_memory_bytes: Option<u64>,
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
impl BenchmarkResult {
    /// Create a new result with the given samples.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        backend: impl Into<String>,
        samples: Vec<Duration>,
        elements_per_iter: u64,
    ) -> Self {
        Self {
            name: name.into(),
            backend: backend.into(),
            samples,
            elements_per_iter,
            peak_memory_bytes: None,
        }
    }

    /// Attach optional memory usage information.
    #[must_use]
    pub const fn with_memory(mut self, bytes: u64) -> Self {
        self.peak_memory_bytes = Some(bytes);
        self
    }

    /// Number of collected samples.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.samples.len()
    }

    /// Arithmetic mean of sample durations.
    #[must_use]
    pub fn mean(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.samples.iter().sum();
        total / self.samples.len() as u32
    }

    /// Population standard deviation of sample durations.
    #[must_use]
    pub fn std_dev(&self) -> Duration {
        if self.samples.len() < 2 {
            return Duration::ZERO;
        }
        let mean_ns = self.mean().as_nanos() as f64;
        let variance = self
            .samples
            .iter()
            .map(|s| {
                let diff = s.as_nanos() as f64 - mean_ns;
                diff * diff
            })
            .sum::<f64>()
            / self.samples.len() as f64;
        Duration::from_nanos(variance.sqrt() as u64)
    }

    /// Minimum sample duration.
    #[must_use]
    pub fn min(&self) -> Duration {
        self.samples.iter().copied().min().unwrap_or(Duration::ZERO)
    }

    /// Maximum sample duration.
    #[must_use]
    pub fn max(&self) -> Duration {
        self.samples.iter().copied().max().unwrap_or(Duration::ZERO)
    }

    /// Percentile (0–100) of sample durations. Uses nearest-rank method.
    #[must_use]
    pub fn percentile(&self, p: f64) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted: Vec<Duration> = self.samples.clone();
        sorted.sort();
        let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Throughput in elements per second, based on the mean duration.
    #[must_use]
    pub fn throughput_eps(&self) -> f64 {
        let mean = self.mean();
        if mean.is_zero() {
            return 0.0;
        }
        self.elements_per_iter as f64 / mean.as_secs_f64()
    }
}

// ── BenchmarkRunner ─────────────────────────────────────────────────────────

/// Runs a closure N times and collects timing samples into a
/// [`BenchmarkResult`].
pub struct BenchmarkRunner {
    /// Number of iterations to collect.
    pub iterations: usize,
    /// Number of warm-up iterations (not recorded).
    pub warmup: usize,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self { iterations: 100, warmup: 10 }
    }
}

impl BenchmarkRunner {
    /// Create a runner with explicit iteration counts.
    #[must_use]
    pub const fn new(iterations: usize, warmup: usize) -> Self {
        Self { iterations, warmup }
    }

    /// Run `f` and collect timing samples.
    pub fn run<F>(
        &self,
        name: impl Into<String>,
        backend: impl Into<String>,
        elements_per_iter: u64,
        mut f: F,
    ) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warm-up phase
        for _ in 0..self.warmup {
            f();
        }

        // Measurement phase
        let mut samples = Vec::with_capacity(self.iterations);
        for _ in 0..self.iterations {
            let start = std::time::Instant::now();
            f();
            samples.push(start.elapsed());
        }

        BenchmarkResult::new(name, backend, samples, elements_per_iter)
    }
}

// ── BenchmarkComparison ─────────────────────────────────────────────────────

/// Collects results from multiple backends for the same benchmark and
/// produces a cross-backend comparison table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// Label for the comparison (e.g. "matmul 256×256").
    pub label: String,
    /// Results from each backend being compared.
    pub results: Vec<BenchmarkResult>,
}

#[allow(clippy::cast_precision_loss)]
impl BenchmarkComparison {
    /// Create a new comparison with a label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self { label: label.into(), results: Vec::new() }
    }

    /// Add a result to this comparison.
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// The fastest backend name (by mean time), or `None` if empty.
    #[must_use]
    pub fn fastest_backend(&self) -> Option<&str> {
        self.results.iter().min_by_key(|r| r.mean()).map(|r| r.backend.as_str())
    }

    /// Speedup of the fastest backend relative to each other backend.
    /// Returns `(backend, speedup_factor)` pairs.
    #[must_use]
    pub fn speedups(&self) -> Vec<(&str, f64)> {
        let Some(fastest) = self.results.iter().min_by_key(|r| r.mean()) else {
            return Vec::new();
        };
        let fastest_ns = fastest.mean().as_nanos() as f64;
        if fastest_ns == 0.0 {
            return Vec::new();
        }
        self.results
            .iter()
            .map(|r| {
                let ratio = r.mean().as_nanos() as f64 / fastest_ns;
                (r.backend.as_str(), ratio)
            })
            .collect()
    }
}

impl fmt::Display for BenchmarkComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {} ===", self.label)?;
        writeln!(
            f,
            "{:<12} {:>12} {:>12} {:>12} {:>12}",
            "Backend", "Mean", "Std Dev", "P50", "Throughput"
        )?;
        writeln!(f, "{}", "-".repeat(64))?;
        for r in &self.results {
            writeln!(
                f,
                "{:<12} {:>12} {:>12} {:>12} {:>10.0} e/s",
                r.backend,
                format_duration(r.mean()),
                format_duration(r.std_dev()),
                format_duration(r.percentile(50.0)),
                r.throughput_eps(),
            )?;
        }
        if let Some(fastest) = self.fastest_backend() {
            writeln!(f, "Fastest: {fastest}")?;
        }
        Ok(())
    }
}

// ── format helpers ──────────────────────────────────────────────────────────

/// Format a single [`BenchmarkResult`] as a human-readable report.
#[must_use]
pub fn format_benchmark_report(result: &BenchmarkResult) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Benchmark: {} (backend: {})", result.name, result.backend);
    let _ = writeln!(out, "  Samples:    {}", result.count());
    let _ = writeln!(out, "  Mean:       {}", format_duration(result.mean()));
    let _ = writeln!(out, "  Std Dev:    {}", format_duration(result.std_dev()));
    let _ = writeln!(out, "  Min:        {}", format_duration(result.min()));
    let _ = writeln!(out, "  Max:        {}", format_duration(result.max()));
    let _ = writeln!(out, "  P50:        {}", format_duration(result.percentile(50.0)));
    let _ = writeln!(out, "  P95:        {}", format_duration(result.percentile(95.0)));
    let _ = writeln!(out, "  P99:        {}", format_duration(result.percentile(99.0)));
    let _ = writeln!(out, "  Throughput: {:.2} elements/s", result.throughput_eps());
    if let Some(mem) = result.peak_memory_bytes {
        let _ = writeln!(out, "  Peak Mem:   {mem} bytes");
    }
    out
}

/// Format a `Duration` in human-friendly units.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos();
    if nanos < 1_000 {
        format!("{nanos} ns")
    } else if nanos < 1_000_000 {
        format!("{:.2} µs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.2} ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.3} s", d.as_secs_f64())
    }
}
