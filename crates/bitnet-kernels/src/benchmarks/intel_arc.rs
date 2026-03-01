//! Intel Arc A770 benchmark definitions and test harness.
//!
//! Provides benchmark scenarios for matmul, softmax, quantized ops,
//! and layer normalization targeting Intel Arc A770 (16 GB, 32 Xe-cores).
//! Includes a CPU reference runner that measures real elapsed time.

use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Benchmark configuration controlling iterations, warmup, and problem sizes.
#[derive(Debug, Clone)]
pub struct ArcBenchConfig {
    /// Number of warmup iterations (excluded from measurements).
    pub warmup_iterations: usize,
    /// Number of timed iterations.
    pub bench_iterations: usize,
    /// Problem sizes for matrix dimensions.
    pub problem_sizes: Vec<usize>,
    /// Label for the configuration.
    pub label: String,
}

impl ArcBenchConfig {
    /// Create a new configuration with the given iteration counts.
    pub fn new(warmup: usize, bench: usize) -> Self {
        Self {
            warmup_iterations: warmup,
            bench_iterations: bench,
            problem_sizes: vec![128, 1024, 4096],
            label: String::from("default"),
        }
    }

    /// Create a configuration suitable for CI (fewer iterations).
    pub fn ci() -> Self {
        Self {
            warmup_iterations: 1,
            bench_iterations: 5,
            problem_sizes: vec![128, 512],
            label: String::from("ci"),
        }
    }

    /// Create a full benchmark configuration.
    pub fn full() -> Self {
        Self {
            warmup_iterations: 10,
            bench_iterations: 100,
            problem_sizes: vec![128, 256, 512, 1024, 2048, 4096],
            label: String::from("full"),
        }
    }

    /// Set custom problem sizes.
    pub fn with_problem_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.problem_sizes = sizes;
        self
    }

    /// Set a label for this config.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Validate that the configuration is sane.
    pub fn validate(&self) -> Result<(), String> {
        if self.bench_iterations == 0 {
            return Err("bench_iterations must be > 0".into());
        }
        if self.problem_sizes.is_empty() {
            return Err("problem_sizes must not be empty".into());
        }
        for &s in &self.problem_sizes {
            if s == 0 {
                return Err("problem sizes must be > 0".into());
            }
        }
        Ok(())
    }
}

impl Default for ArcBenchConfig {
    fn default() -> Self {
        Self::new(5, 50)
    }
}

// ---------------------------------------------------------------------------
// Benchmark scenarios
// ---------------------------------------------------------------------------

/// Matrix multiplication benchmark scenarios.
#[derive(Debug, Clone)]
pub struct MatmulBenchmark {
    /// Rows of the left matrix.
    pub m: usize,
    /// Inner dimension.
    pub k: usize,
    /// Columns of the right matrix.
    pub n: usize,
    /// Human-readable label.
    pub label: String,
}

impl MatmulBenchmark {
    pub fn new(m: usize, k: usize, n: usize, label: impl Into<String>) -> Self {
        Self { m, k, n, label: label.into() }
    }

    /// Small matmul scenario (128×128).
    pub fn small() -> Self {
        Self::new(128, 128, 128, "matmul-small-128x128")
    }

    /// Medium matmul scenario (1024×1024).
    pub fn medium() -> Self {
        Self::new(1024, 1024, 1024, "matmul-medium-1024x1024")
    }

    /// Large matmul scenario (4096×4096).
    pub fn large() -> Self {
        Self::new(4096, 4096, 4096, "matmul-large-4096x4096")
    }

    /// All standard matmul scenarios.
    pub fn all() -> Vec<Self> {
        vec![Self::small(), Self::medium(), Self::large()]
    }

    /// Total number of FP operations (2*M*N*K for matmul).
    pub fn flop_count(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
    }

    /// Run on CPU and return elapsed time for one iteration.
    pub fn run_cpu_reference(&self) -> Duration {
        let a = vec![1.0f32; self.m * self.k];
        let b = vec![1.0f32; self.k * self.n];
        let mut c = vec![0.0f32; self.m * self.n];

        let start = Instant::now();
        naive_matmul(&a, &b, &mut c, self.m, self.k, self.n);
        start.elapsed()
    }
}

/// Softmax benchmark scenarios targeting common vocabulary sizes.
#[derive(Debug, Clone)]
pub struct SoftmaxBenchmark {
    /// Vocabulary / logit vector length.
    pub vocab_size: usize,
    /// Batch size (number of independent softmax calls).
    pub batch_size: usize,
    /// Human-readable label.
    pub label: String,
}

impl SoftmaxBenchmark {
    pub fn new(vocab_size: usize, batch_size: usize, label: impl Into<String>) -> Self {
        Self { vocab_size, batch_size, label: label.into() }
    }

    /// LLaMA-style vocabulary (32 000).
    pub fn llama() -> Self {
        Self::new(32_000, 1, "softmax-llama-32k")
    }

    /// GPT-2 vocabulary (50 257).
    pub fn gpt2() -> Self {
        Self::new(50_257, 1, "softmax-gpt2-50k")
    }

    /// Large vocabulary (128 256).
    pub fn large_vocab() -> Self {
        Self::new(128_256, 1, "softmax-large-128k")
    }

    /// All standard softmax scenarios.
    pub fn all() -> Vec<Self> {
        vec![Self::llama(), Self::gpt2(), Self::large_vocab()]
    }

    /// Run on CPU and return elapsed time.
    pub fn run_cpu_reference(&self) -> Duration {
        let logits = vec![0.5f32; self.vocab_size * self.batch_size];
        let mut out = vec![0.0f32; logits.len()];

        let start = Instant::now();
        for b in 0..self.batch_size {
            let offset = b * self.vocab_size;
            let slice = &logits[offset..offset + self.vocab_size];
            let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (i, &v) in slice.iter().enumerate() {
                let e = (v - max_val).exp();
                out[offset + i] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for i in 0..self.vocab_size {
                out[offset + i] *= inv_sum;
            }
        }
        let _ = &out; // prevent optimisation
        start.elapsed()
    }
}

/// Quantized I2_S dequantize + matvec benchmark at BitNet model dimensions.
#[derive(Debug, Clone)]
pub struct QuantizedBenchmark {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Intermediate (FFN) dimension.
    pub intermediate_dim: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Human-readable label.
    pub label: String,
}

impl QuantizedBenchmark {
    pub fn new(
        hidden: usize,
        intermediate: usize,
        seq_len: usize,
        label: impl Into<String>,
    ) -> Self {
        Self {
            hidden_dim: hidden,
            intermediate_dim: intermediate,
            seq_len,
            label: label.into(),
        }
    }

    /// BitNet 2B model dimensions (hidden=2048, intermediate=5504).
    pub fn bitnet_2b() -> Self {
        Self::new(2048, 5504, 1, "quant-bitnet-2b")
    }

    /// BitNet 700M model dimensions (hidden=1024, intermediate=2816).
    pub fn bitnet_700m() -> Self {
        Self::new(1024, 2816, 1, "quant-bitnet-700m")
    }

    /// All standard quantized scenarios.
    pub fn all() -> Vec<Self> {
        vec![Self::bitnet_700m(), Self::bitnet_2b()]
    }

    /// Run a CPU reference: simulated dequant + matvec.
    pub fn run_cpu_reference(&self) -> Duration {
        let packed_len = (self.hidden_dim * self.intermediate_dim + 3) / 4;
        let packed = vec![0xAAu8; packed_len];
        let scales = vec![1.0f32; self.intermediate_dim];
        let input = vec![1.0f32; self.hidden_dim * self.seq_len];
        let mut output = vec![0.0f32; self.intermediate_dim * self.seq_len];

        let start = Instant::now();
        cpu_dequant_matvec(&packed, &scales, &input, &mut output, self.hidden_dim, self.intermediate_dim, self.seq_len);
        start.elapsed()
    }
}

/// LayerNorm / RMSNorm benchmark at various hidden sizes.
#[derive(Debug, Clone)]
pub struct LayerNormBenchmark {
    /// Hidden dimension (vector length).
    pub hidden_size: usize,
    /// Number of vectors to normalise.
    pub batch_size: usize,
    /// Whether to use RMSNorm instead of full LayerNorm.
    pub rms_norm: bool,
    /// Human-readable label.
    pub label: String,
}

impl LayerNormBenchmark {
    pub fn new(hidden: usize, batch: usize, rms: bool, label: impl Into<String>) -> Self {
        Self { hidden_size: hidden, batch_size: batch, rms_norm: rms, label: label.into() }
    }

    /// Small model hidden size (768).
    pub fn small() -> Self {
        Self::new(768, 32, false, "layernorm-768")
    }

    /// Medium model hidden size (2048).
    pub fn medium() -> Self {
        Self::new(2048, 32, true, "rmsnorm-2048")
    }

    /// Large model hidden size (4096).
    pub fn large() -> Self {
        Self::new(4096, 32, true, "rmsnorm-4096")
    }

    /// All standard layer-norm scenarios.
    pub fn all() -> Vec<Self> {
        vec![Self::small(), Self::medium(), Self::large()]
    }

    /// Run on CPU and return elapsed time.
    pub fn run_cpu_reference(&self) -> Duration {
        let data = vec![1.0f32; self.hidden_size * self.batch_size];
        let mut out = vec![0.0f32; data.len()];
        let eps = 1e-5f32;

        let start = Instant::now();
        for b in 0..self.batch_size {
            let offset = b * self.hidden_size;
            let slice = &data[offset..offset + self.hidden_size];
            if self.rms_norm {
                let mean_sq: f32 = slice.iter().map(|x| x * x).sum::<f32>() / self.hidden_size as f32;
                let inv = 1.0 / (mean_sq + eps).sqrt();
                for (i, &v) in slice.iter().enumerate() {
                    out[offset + i] = v * inv;
                }
            } else {
                let mean: f32 = slice.iter().sum::<f32>() / self.hidden_size as f32;
                let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / self.hidden_size as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for (i, &v) in slice.iter().enumerate() {
                    out[offset + i] = (v - mean) * inv;
                }
            }
        }
        let _ = &out;
        start.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Timing result with statistics
// ---------------------------------------------------------------------------

/// Timing result with statistical aggregates.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Name of the benchmark.
    pub name: String,
    /// All measured durations (sorted).
    pub samples: Vec<Duration>,
}

impl BenchResult {
    /// Create a result from raw sample durations.
    pub fn new(name: impl Into<String>, mut samples: Vec<Duration>) -> Self {
        samples.sort();
        Self { name: name.into(), samples }
    }

    /// Number of samples.
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Arithmetic mean duration.
    pub fn mean(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.samples.iter().sum();
        total / self.samples.len() as u32
    }

    /// Median duration.
    pub fn median(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mid = self.samples.len() / 2;
        if self.samples.len() % 2 == 0 {
            (self.samples[mid - 1] + self.samples[mid]) / 2
        } else {
            self.samples[mid]
        }
    }

    /// Percentile duration (0–100).
    pub fn percentile(&self, p: f64) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let idx = ((p / 100.0) * (self.samples.len() - 1) as f64).round() as usize;
        self.samples[idx.min(self.samples.len() - 1)]
    }

    /// 95th-percentile duration.
    pub fn p95(&self) -> Duration {
        self.percentile(95.0)
    }

    /// 99th-percentile duration.
    pub fn p99(&self) -> Duration {
        self.percentile(99.0)
    }

    /// Minimum sample.
    pub fn min(&self) -> Duration {
        self.samples.first().copied().unwrap_or(Duration::ZERO)
    }

    /// Maximum sample.
    pub fn max(&self) -> Duration {
        self.samples.last().copied().unwrap_or(Duration::ZERO)
    }

    /// Standard deviation in seconds.
    pub fn std_dev_secs(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean_s = self.mean().as_secs_f64();
        let var: f64 = self
            .samples
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean_s;
                diff * diff
            })
            .sum::<f64>()
            / (self.samples.len() - 1) as f64;
        var.sqrt()
    }

    /// Coefficient of variation (std_dev / mean).
    pub fn cv(&self) -> f64 {
        let m = self.mean().as_secs_f64();
        if m == 0.0 {
            return 0.0;
        }
        self.std_dev_secs() / m
    }
}

impl fmt::Display for BenchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: mean={:.3?} median={:.3?} p95={:.3?} p99={:.3?} stddev={:.6}s (n={})",
            self.name,
            self.mean(),
            self.median(),
            self.p95(),
            self.p99(),
            self.std_dev_secs(),
            self.count(),
        )
    }
}

// ---------------------------------------------------------------------------
// Performance profile
// ---------------------------------------------------------------------------

/// Expected performance characteristics for Intel Arc A770.
#[derive(Debug, Clone)]
pub struct ArcPerformanceProfile {
    /// GPU name.
    pub name: &'static str,
    /// VRAM in bytes.
    pub vram_bytes: u64,
    /// Number of Xe-cores.
    pub xe_cores: u32,
    /// Peak FP32 TFLOPS.
    pub peak_fp32_tflops: f64,
    /// Peak INT8 TOPS.
    pub peak_int8_tops: f64,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// PCI-Express generation.
    pub pcie_gen: u32,
    /// PCI-Express lane count.
    pub pcie_lanes: u32,
}

impl ArcPerformanceProfile {
    /// Arc A770 16 GB reference profile.
    pub fn a770_16gb() -> Self {
        Self {
            name: "Intel Arc A770 16GB",
            vram_bytes: 16 * 1024 * 1024 * 1024,
            xe_cores: 32,
            peak_fp32_tflops: 19.66,
            peak_int8_tops: 39.32,
            memory_bandwidth_gbps: 560.0,
            pcie_gen: 4,
            pcie_lanes: 16,
        }
    }

    /// Arc A770 8 GB variant.
    pub fn a770_8gb() -> Self {
        Self {
            name: "Intel Arc A770 8GB",
            vram_bytes: 8 * 1024 * 1024 * 1024,
            xe_cores: 32,
            peak_fp32_tflops: 19.66,
            peak_int8_tops: 39.32,
            memory_bandwidth_gbps: 560.0,
            pcie_gen: 4,
            pcie_lanes: 16,
        }
    }

    /// VRAM in GiB.
    pub fn vram_gib(&self) -> f64 {
        self.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Theoretical peak FP32 GFLOPS.
    pub fn peak_fp32_gflops(&self) -> f64 {
        self.peak_fp32_tflops * 1000.0
    }

    /// Theoretical PCIe bandwidth in GB/s (uni-directional).
    pub fn pcie_bandwidth_gbps(&self) -> f64 {
        // PCIe 4.0 ≈ 1.969 GB/s per lane
        self.pcie_lanes as f64 * 1.969
    }

    /// Compute efficiency given observed GFLOPS.
    pub fn compute_efficiency(&self, observed_gflops: f64) -> f64 {
        observed_gflops / self.peak_fp32_gflops()
    }

    /// Memory bandwidth efficiency given observed GB/s.
    pub fn memory_efficiency(&self, observed_gbps: f64) -> f64 {
        observed_gbps / self.memory_bandwidth_gbps
    }
}

impl Default for ArcPerformanceProfile {
    fn default() -> Self {
        Self::a770_16gb()
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Runs benchmark scenarios and collects [`BenchResult`]s.
#[derive(Debug, Clone)]
pub struct BenchRunner {
    config: ArcBenchConfig,
    results: Vec<BenchResult>,
}

impl BenchRunner {
    /// Create a new runner with the given config.
    pub fn new(config: ArcBenchConfig) -> Self {
        Self { config, results: Vec::new() }
    }

    /// Access the config.
    pub fn config(&self) -> &ArcBenchConfig {
        &self.config
    }

    /// Run a matmul benchmark and record the result.
    pub fn run_matmul(&mut self, bench: &MatmulBenchmark) {
        let samples = self.timed_run(|_| bench.run_cpu_reference());
        self.results.push(BenchResult::new(&bench.label, samples));
    }

    /// Run a softmax benchmark and record the result.
    pub fn run_softmax(&mut self, bench: &SoftmaxBenchmark) {
        let samples = self.timed_run(|_| bench.run_cpu_reference());
        self.results.push(BenchResult::new(&bench.label, samples));
    }

    /// Run a quantized-ops benchmark and record the result.
    pub fn run_quantized(&mut self, bench: &QuantizedBenchmark) {
        let samples = self.timed_run(|_| bench.run_cpu_reference());
        self.results.push(BenchResult::new(&bench.label, samples));
    }

    /// Run a layer-norm benchmark and record the result.
    pub fn run_layer_norm(&mut self, bench: &LayerNormBenchmark) {
        let samples = self.timed_run(|_| bench.run_cpu_reference());
        self.results.push(BenchResult::new(&bench.label, samples));
    }

    /// Collected results so far.
    pub fn results(&self) -> &[BenchResult] {
        &self.results
    }

    /// Clear all results.
    pub fn clear(&mut self) {
        self.results.clear();
    }

    // Internal: warmup + timed iterations.
    fn timed_run<F>(&self, f: F) -> Vec<Duration>
    where
        F: Fn(usize) -> Duration,
    {
        // Warmup
        for i in 0..self.config.warmup_iterations {
            let _ = f(i);
        }
        // Timed
        (0..self.config.bench_iterations).map(|i| f(i)).collect()
    }
}

// ---------------------------------------------------------------------------
// Reporter
// ---------------------------------------------------------------------------

/// Formats benchmark results as markdown or JSON.
#[derive(Debug, Clone)]
pub struct BenchReporter {
    profile: ArcPerformanceProfile,
}

impl BenchReporter {
    /// Create a reporter for the given performance profile.
    pub fn new(profile: ArcPerformanceProfile) -> Self {
        Self { profile }
    }

    /// Render results as a markdown table.
    pub fn to_markdown(&self, results: &[BenchResult]) -> String {
        let mut out = String::new();
        out.push_str(&format!("# Benchmark Results — {}\n\n", self.profile.name));
        out.push_str("| Benchmark | Mean | Median | P95 | P99 | Std Dev | N |\n");
        out.push_str("|-----------|------|--------|-----|-----|---------|---|\n");
        for r in results {
            out.push_str(&format!(
                "| {} | {:.3?} | {:.3?} | {:.3?} | {:.3?} | {:.6}s | {} |\n",
                r.name,
                r.mean(),
                r.median(),
                r.p95(),
                r.p99(),
                r.std_dev_secs(),
                r.count(),
            ));
        }
        out
    }

    /// Render results as a JSON string.
    pub fn to_json(&self, results: &[BenchResult]) -> String {
        let mut entries: Vec<String> = Vec::new();
        for r in results {
            entries.push(format!(
                concat!(
                    "    {{\n",
                    "      \"name\": \"{}\",\n",
                    "      \"mean_ns\": {},\n",
                    "      \"median_ns\": {},\n",
                    "      \"p95_ns\": {},\n",
                    "      \"p99_ns\": {},\n",
                    "      \"std_dev_s\": {:.9},\n",
                    "      \"samples\": {}\n",
                    "    }}"
                ),
                r.name,
                r.mean().as_nanos(),
                r.median().as_nanos(),
                r.p95().as_nanos(),
                r.p99().as_nanos(),
                r.std_dev_secs(),
                r.count(),
            ));
        }
        format!(
            "{{\n  \"device\": \"{}\",\n  \"results\": [\n{}\n  ]\n}}",
            self.profile.name,
            entries.join(",\n"),
        )
    }
}

// ---------------------------------------------------------------------------
// Complete benchmark suite
// ---------------------------------------------------------------------------

/// Complete benchmark suite for Intel Arc A770 validation.
///
/// Combines all scenario types and runs them with a shared config.
#[derive(Debug, Clone)]
pub struct IntelArcBenchSuite {
    config: ArcBenchConfig,
    profile: ArcPerformanceProfile,
    matmul_benchmarks: Vec<MatmulBenchmark>,
    softmax_benchmarks: Vec<SoftmaxBenchmark>,
    quantized_benchmarks: Vec<QuantizedBenchmark>,
    layer_norm_benchmarks: Vec<LayerNormBenchmark>,
}

impl IntelArcBenchSuite {
    /// Create a suite with defaults for A770 16 GB.
    pub fn new(config: ArcBenchConfig) -> Self {
        Self {
            config,
            profile: ArcPerformanceProfile::a770_16gb(),
            matmul_benchmarks: MatmulBenchmark::all(),
            softmax_benchmarks: SoftmaxBenchmark::all(),
            quantized_benchmarks: QuantizedBenchmark::all(),
            layer_norm_benchmarks: LayerNormBenchmark::all(),
        }
    }

    /// Override the performance profile.
    pub fn with_profile(mut self, profile: ArcPerformanceProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Override matmul benchmarks.
    pub fn with_matmul(mut self, benchmarks: Vec<MatmulBenchmark>) -> Self {
        self.matmul_benchmarks = benchmarks;
        self
    }

    /// Override softmax benchmarks.
    pub fn with_softmax(mut self, benchmarks: Vec<SoftmaxBenchmark>) -> Self {
        self.softmax_benchmarks = benchmarks;
        self
    }

    /// Override quantized benchmarks.
    pub fn with_quantized(mut self, benchmarks: Vec<QuantizedBenchmark>) -> Self {
        self.quantized_benchmarks = benchmarks;
        self
    }

    /// Override layer-norm benchmarks.
    pub fn with_layer_norm(mut self, benchmarks: Vec<LayerNormBenchmark>) -> Self {
        self.layer_norm_benchmarks = benchmarks;
        self
    }

    /// Access the performance profile.
    pub fn profile(&self) -> &ArcPerformanceProfile {
        &self.profile
    }

    /// Total number of benchmark scenarios.
    pub fn scenario_count(&self) -> usize {
        self.matmul_benchmarks.len()
            + self.softmax_benchmarks.len()
            + self.quantized_benchmarks.len()
            + self.layer_norm_benchmarks.len()
    }

    /// Run the full suite and return all results.
    pub fn run(&self) -> Vec<BenchResult> {
        let mut runner = BenchRunner::new(self.config.clone());
        for b in &self.matmul_benchmarks {
            runner.run_matmul(b);
        }
        for b in &self.softmax_benchmarks {
            runner.run_softmax(b);
        }
        for b in &self.quantized_benchmarks {
            runner.run_quantized(b);
        }
        for b in &self.layer_norm_benchmarks {
            runner.run_layer_norm(b);
        }
        runner.results().to_vec()
    }

    /// Run the suite and produce a markdown report.
    pub fn run_and_report_markdown(&self) -> String {
        let results = self.run();
        let reporter = BenchReporter::new(self.profile.clone());
        reporter.to_markdown(&results)
    }

    /// Run the suite and produce a JSON report.
    pub fn run_and_report_json(&self) -> String {
        let results = self.run();
        let reporter = BenchReporter::new(self.profile.clone());
        reporter.to_json(&results)
    }
}

// ---------------------------------------------------------------------------
// CPU reference helpers (used by benchmark scenarios)
// ---------------------------------------------------------------------------

/// Naive row-major matmul: C = A × B.
fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Simulated I2_S dequant + matvec on CPU.
fn cpu_dequant_matvec(
    packed: &[u8],
    scales: &[f32],
    input: &[f32],
    output: &mut [f32],
    hidden: usize,
    intermediate: usize,
    seq_len: usize,
) {
    // Simulate dequantization + matrix-vector multiply
    for s in 0..seq_len {
        for row in 0..intermediate {
            let mut acc = 0.0f32;
            for col in 0..hidden {
                let packed_idx = (row * hidden + col) / 4;
                let shift = ((row * hidden + col) % 4) * 2;
                let raw = (packed[packed_idx % packed.len()] >> shift) & 0x03;
                let weight = (raw as f32 - 1.0) * scales[row % scales.len()];
                acc += weight * input[s * hidden + col];
            }
            output[s * intermediate + row] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ArcBenchConfig tests -----------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = ArcBenchConfig::default();
        assert_eq!(cfg.warmup_iterations, 5);
        assert_eq!(cfg.bench_iterations, 50);
        assert_eq!(cfg.problem_sizes, vec![128, 1024, 4096]);
    }

    #[test]
    fn test_config_ci() {
        let cfg = ArcBenchConfig::ci();
        assert_eq!(cfg.warmup_iterations, 1);
        assert_eq!(cfg.bench_iterations, 5);
        assert_eq!(cfg.label, "ci");
    }

    #[test]
    fn test_config_full() {
        let cfg = ArcBenchConfig::full();
        assert_eq!(cfg.warmup_iterations, 10);
        assert_eq!(cfg.bench_iterations, 100);
        assert_eq!(cfg.problem_sizes.len(), 6);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = ArcBenchConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_iterations() {
        let cfg = ArcBenchConfig::new(0, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_empty_sizes() {
        let cfg = ArcBenchConfig::default().with_problem_sizes(vec![]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_size() {
        let cfg = ArcBenchConfig::default().with_problem_sizes(vec![128, 0, 256]);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_with_label() {
        let cfg = ArcBenchConfig::default().with_label("custom");
        assert_eq!(cfg.label, "custom");
    }

    // -- MatmulBenchmark tests ----------------------------------------------

    #[test]
    fn test_matmul_small() {
        let b = MatmulBenchmark::small();
        assert_eq!(b.m, 128);
        assert_eq!(b.k, 128);
        assert_eq!(b.n, 128);
    }

    #[test]
    fn test_matmul_medium() {
        let b = MatmulBenchmark::medium();
        assert_eq!(b.m, 1024);
    }

    #[test]
    fn test_matmul_large() {
        let b = MatmulBenchmark::large();
        assert_eq!(b.m, 4096);
    }

    #[test]
    fn test_matmul_all_count() {
        assert_eq!(MatmulBenchmark::all().len(), 3);
    }

    #[test]
    fn test_matmul_flop_count() {
        let b = MatmulBenchmark::small();
        // 2 * 128^3 = 4_194_304
        assert_eq!(b.flop_count(), 2 * 128u64.pow(3));
    }

    #[test]
    fn test_matmul_cpu_reference_runs() {
        let b = MatmulBenchmark::small();
        let d = b.run_cpu_reference();
        assert!(d.as_nanos() > 0, "matmul should take non-zero time");
    }

    // -- SoftmaxBenchmark tests ---------------------------------------------

    #[test]
    fn test_softmax_llama() {
        let b = SoftmaxBenchmark::llama();
        assert_eq!(b.vocab_size, 32_000);
    }

    #[test]
    fn test_softmax_gpt2() {
        let b = SoftmaxBenchmark::gpt2();
        assert_eq!(b.vocab_size, 50_257);
    }

    #[test]
    fn test_softmax_large() {
        let b = SoftmaxBenchmark::large_vocab();
        assert_eq!(b.vocab_size, 128_256);
    }

    #[test]
    fn test_softmax_all_count() {
        assert_eq!(SoftmaxBenchmark::all().len(), 3);
    }

    #[test]
    fn test_softmax_cpu_reference_runs() {
        let b = SoftmaxBenchmark::llama();
        let d = b.run_cpu_reference();
        assert!(d.as_nanos() > 0);
    }

    // -- QuantizedBenchmark tests -------------------------------------------

    #[test]
    fn test_quant_bitnet_2b() {
        let b = QuantizedBenchmark::bitnet_2b();
        assert_eq!(b.hidden_dim, 2048);
        assert_eq!(b.intermediate_dim, 5504);
    }

    #[test]
    fn test_quant_bitnet_700m() {
        let b = QuantizedBenchmark::bitnet_700m();
        assert_eq!(b.hidden_dim, 1024);
    }

    #[test]
    fn test_quant_all_count() {
        assert_eq!(QuantizedBenchmark::all().len(), 2);
    }

    #[test]
    fn test_quant_cpu_reference_runs() {
        let b = QuantizedBenchmark::bitnet_700m();
        let d = b.run_cpu_reference();
        assert!(d.as_nanos() > 0);
    }

    // -- LayerNormBenchmark tests -------------------------------------------

    #[test]
    fn test_layernorm_small() {
        let b = LayerNormBenchmark::small();
        assert_eq!(b.hidden_size, 768);
        assert!(!b.rms_norm);
    }

    #[test]
    fn test_layernorm_medium() {
        let b = LayerNormBenchmark::medium();
        assert_eq!(b.hidden_size, 2048);
        assert!(b.rms_norm);
    }

    #[test]
    fn test_layernorm_large() {
        let b = LayerNormBenchmark::large();
        assert_eq!(b.hidden_size, 4096);
    }

    #[test]
    fn test_layernorm_all_count() {
        assert_eq!(LayerNormBenchmark::all().len(), 3);
    }

    #[test]
    fn test_layernorm_cpu_reference_runs() {
        let b = LayerNormBenchmark::small();
        let d = b.run_cpu_reference();
        assert!(d.as_nanos() > 0);
    }

    // -- BenchResult tests --------------------------------------------------

    #[test]
    fn test_result_empty() {
        let r = BenchResult::new("empty", vec![]);
        assert_eq!(r.count(), 0);
        assert_eq!(r.mean(), Duration::ZERO);
        assert_eq!(r.median(), Duration::ZERO);
        assert_eq!(r.p95(), Duration::ZERO);
        assert_eq!(r.std_dev_secs(), 0.0);
    }

    #[test]
    fn test_result_single_sample() {
        let r = BenchResult::new("one", vec![Duration::from_millis(10)]);
        assert_eq!(r.count(), 1);
        assert_eq!(r.mean(), Duration::from_millis(10));
        assert_eq!(r.median(), Duration::from_millis(10));
        assert_eq!(r.std_dev_secs(), 0.0);
    }

    #[test]
    fn test_result_mean() {
        let r = BenchResult::new(
            "mean",
            vec![Duration::from_millis(10), Duration::from_millis(20), Duration::from_millis(30)],
        );
        assert_eq!(r.mean(), Duration::from_millis(20));
    }

    #[test]
    fn test_result_median_odd() {
        let r = BenchResult::new(
            "med",
            vec![Duration::from_millis(5), Duration::from_millis(10), Duration::from_millis(100)],
        );
        assert_eq!(r.median(), Duration::from_millis(10));
    }

    #[test]
    fn test_result_median_even() {
        let r = BenchResult::new(
            "med_even",
            vec![Duration::from_millis(10), Duration::from_millis(20)],
        );
        assert_eq!(r.median(), Duration::from_millis(15));
    }

    #[test]
    fn test_result_percentile_p95() {
        let samples: Vec<Duration> = (1..=100).map(|i| Duration::from_millis(i)).collect();
        let r = BenchResult::new("pct", samples);
        assert_eq!(r.p95(), Duration::from_millis(95));
    }

    #[test]
    fn test_result_percentile_p99() {
        let samples: Vec<Duration> = (1..=100).map(|i| Duration::from_millis(i)).collect();
        let r = BenchResult::new("pct99", samples);
        assert_eq!(r.p99(), Duration::from_millis(99));
    }

    #[test]
    fn test_result_min_max() {
        let r = BenchResult::new(
            "mm",
            vec![Duration::from_millis(3), Duration::from_millis(7), Duration::from_millis(1)],
        );
        assert_eq!(r.min(), Duration::from_millis(1));
        assert_eq!(r.max(), Duration::from_millis(7));
    }

    #[test]
    fn test_result_std_dev() {
        let r = BenchResult::new(
            "sd",
            vec![Duration::from_millis(10), Duration::from_millis(10), Duration::from_millis(10)],
        );
        assert!(r.std_dev_secs() < 1e-9, "identical samples → zero stddev");
    }

    #[test]
    fn test_result_cv() {
        let r = BenchResult::new(
            "cv",
            vec![Duration::from_millis(10), Duration::from_millis(10)],
        );
        assert!(r.cv() < 1e-9);
    }

    #[test]
    fn test_result_display() {
        let r = BenchResult::new("disp", vec![Duration::from_millis(5)]);
        let s = format!("{r}");
        assert!(s.contains("disp"));
        assert!(s.contains("mean="));
    }

    #[test]
    fn test_result_sorted() {
        let r = BenchResult::new(
            "sorted",
            vec![Duration::from_millis(30), Duration::from_millis(10), Duration::from_millis(20)],
        );
        assert_eq!(r.samples[0], Duration::from_millis(10));
        assert_eq!(r.samples[2], Duration::from_millis(30));
    }

    // -- ArcPerformanceProfile tests ----------------------------------------

    #[test]
    fn test_profile_a770_16gb() {
        let p = ArcPerformanceProfile::a770_16gb();
        assert_eq!(p.xe_cores, 32);
        assert_eq!(p.vram_bytes, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_profile_a770_8gb() {
        let p = ArcPerformanceProfile::a770_8gb();
        assert_eq!(p.vram_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_profile_vram_gib() {
        let p = ArcPerformanceProfile::a770_16gb();
        assert!((p.vram_gib() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_profile_peak_gflops() {
        let p = ArcPerformanceProfile::a770_16gb();
        assert!((p.peak_fp32_gflops() - 19660.0).abs() < 1.0);
    }

    #[test]
    fn test_profile_pcie_bandwidth() {
        let p = ArcPerformanceProfile::a770_16gb();
        // 16 lanes × ~1.969 GB/s ≈ 31.5 GB/s
        assert!(p.pcie_bandwidth_gbps() > 30.0);
        assert!(p.pcie_bandwidth_gbps() < 33.0);
    }

    #[test]
    fn test_profile_compute_efficiency() {
        let p = ArcPerformanceProfile::a770_16gb();
        let eff = p.compute_efficiency(9830.0);
        assert!((eff - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_profile_memory_efficiency() {
        let p = ArcPerformanceProfile::a770_16gb();
        let eff = p.memory_efficiency(280.0);
        assert!((eff - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_profile_default() {
        let p = ArcPerformanceProfile::default();
        assert_eq!(p.name, "Intel Arc A770 16GB");
    }

    // -- BenchRunner tests --------------------------------------------------

    #[test]
    fn test_runner_new() {
        let runner = BenchRunner::new(ArcBenchConfig::ci());
        assert!(runner.results().is_empty());
    }

    #[test]
    fn test_runner_run_matmul() {
        let mut runner = BenchRunner::new(ArcBenchConfig::new(0, 2));
        runner.run_matmul(&MatmulBenchmark::small());
        assert_eq!(runner.results().len(), 1);
        assert_eq!(runner.results()[0].count(), 2);
    }

    #[test]
    fn test_runner_run_softmax() {
        let mut runner = BenchRunner::new(ArcBenchConfig::new(0, 2));
        runner.run_softmax(&SoftmaxBenchmark::llama());
        assert_eq!(runner.results().len(), 1);
    }

    #[test]
    fn test_runner_run_quantized() {
        let mut runner = BenchRunner::new(ArcBenchConfig::new(0, 2));
        runner.run_quantized(&QuantizedBenchmark::bitnet_700m());
        assert_eq!(runner.results().len(), 1);
    }

    #[test]
    fn test_runner_run_layer_norm() {
        let mut runner = BenchRunner::new(ArcBenchConfig::new(0, 2));
        runner.run_layer_norm(&LayerNormBenchmark::small());
        assert_eq!(runner.results().len(), 1);
    }

    #[test]
    fn test_runner_clear() {
        let mut runner = BenchRunner::new(ArcBenchConfig::new(0, 1));
        runner.run_softmax(&SoftmaxBenchmark::llama());
        assert_eq!(runner.results().len(), 1);
        runner.clear();
        assert!(runner.results().is_empty());
    }

    #[test]
    fn test_runner_config_accessor() {
        let runner = BenchRunner::new(ArcBenchConfig::ci());
        assert_eq!(runner.config().label, "ci");
    }

    // -- BenchReporter tests ------------------------------------------------

    #[test]
    fn test_reporter_markdown_header() {
        let reporter = BenchReporter::new(ArcPerformanceProfile::a770_16gb());
        let md = reporter.to_markdown(&[]);
        assert!(md.contains("Intel Arc A770 16GB"));
        assert!(md.contains("| Benchmark |"));
    }

    #[test]
    fn test_reporter_markdown_row() {
        let reporter = BenchReporter::new(ArcPerformanceProfile::a770_16gb());
        let results = vec![BenchResult::new("test-op", vec![Duration::from_millis(5)])];
        let md = reporter.to_markdown(&results);
        assert!(md.contains("test-op"));
    }

    #[test]
    fn test_reporter_json_structure() {
        let reporter = BenchReporter::new(ArcPerformanceProfile::a770_16gb());
        let results = vec![BenchResult::new("test-op", vec![Duration::from_millis(5)])];
        let json = reporter.to_json(&results);
        assert!(json.contains("\"device\""));
        assert!(json.contains("\"name\": \"test-op\""));
        assert!(json.contains("\"mean_ns\""));
        assert!(json.contains("\"p95_ns\""));
    }

    #[test]
    fn test_reporter_json_empty() {
        let reporter = BenchReporter::new(ArcPerformanceProfile::a770_16gb());
        let json = reporter.to_json(&[]);
        assert!(json.contains("\"results\": ["));
    }

    // -- IntelArcBenchSuite tests -------------------------------------------

    #[test]
    fn test_suite_scenario_count() {
        let suite = IntelArcBenchSuite::new(ArcBenchConfig::ci());
        // 3 matmul + 3 softmax + 2 quantized + 3 layernorm = 11
        assert_eq!(suite.scenario_count(), 11);
    }

    #[test]
    fn test_suite_profile_accessor() {
        let suite = IntelArcBenchSuite::new(ArcBenchConfig::ci());
        assert_eq!(suite.profile().xe_cores, 32);
    }

    #[test]
    fn test_suite_with_profile() {
        let suite = IntelArcBenchSuite::new(ArcBenchConfig::ci())
            .with_profile(ArcPerformanceProfile::a770_8gb());
        assert_eq!(suite.profile().vram_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_suite_custom_matmul() {
        let suite = IntelArcBenchSuite::new(ArcBenchConfig::ci())
            .with_matmul(vec![MatmulBenchmark::small()]);
        // 1 matmul + 3 softmax + 2 quantized + 3 layernorm = 9
        assert_eq!(suite.scenario_count(), 9);
    }

    #[test]
    fn test_suite_run_returns_results() {
        let suite = IntelArcBenchSuite::new(ArcBenchConfig::new(0, 1))
            .with_matmul(vec![MatmulBenchmark::small()])
            .with_softmax(vec![])
            .with_quantized(vec![])
            .with_layer_norm(vec![]);
        let results = suite.run();
        assert_eq!(results.len(), 1);
        assert!(results[0].mean().as_nanos() > 0);
    }

    #[test]
    fn test_suite_run_and_report_markdown() {
        let suite = IntelArcBenchSuite::new(ArcBenchConfig::new(0, 1))
            .with_matmul(vec![MatmulBenchmark::small()])
            .with_softmax(vec![])
            .with_quantized(vec![])
            .with_layer_norm(vec![]);
        let md = suite.run_and_report_markdown();
        assert!(md.contains("matmul-small-128x128"));
    }

    #[test]
    fn test_suite_run_and_report_json() {
        let suite = IntelArcBenchSuite::new(ArcBenchConfig::new(0, 1))
            .with_matmul(vec![MatmulBenchmark::small()])
            .with_softmax(vec![])
            .with_quantized(vec![])
            .with_layer_norm(vec![]);
        let json = suite.run_and_report_json();
        assert!(json.contains("\"name\": \"matmul-small-128x128\""));
    }

    // -- CPU reference helper tests -----------------------------------------

    #[test]
    fn test_naive_matmul_identity() {
        // 2×2 identity × [1,2,3,4] = [1,2,3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f32; 4];
        naive_matmul(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_naive_matmul_ones() {
        // 2×3 ones × 3×2 ones → each element = 3
        let a = vec![1.0f32; 6];
        let b = vec![1.0f32; 6];
        let mut c = vec![0.0f32; 4];
        naive_matmul(&a, &b, &mut c, 2, 3, 2);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }
}
