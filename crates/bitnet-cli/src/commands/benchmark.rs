//! Benchmarking command implementation

use anyhow::{Context, Result};
use clap::Args;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use bitnet_inference::InferenceEngine;
use bitnet_models::ModelLoader;
use bitnet_tokenizers::{Tokenizer, TokenizerBuilder};
use candle_core::Device;

use crate::config::{CliConfig, invalid_device_message, is_supported_device_label};

use super::receipts::explain_receipt;

const BENCHMARK_DEVICE_HELP: &str = "Device for this legacy benchmark. Use --cuda-benchmark-receipt with cuda/nvidia-rtx-5070-ti-cuda to report governed receipt-backed CUDA benchmark evidence; without a receipt only cpu/auto are supported.";

const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";

const GOVERNED_CUDA_BENCHMARK_ARTIFACTS: &[&str] = &[
    "strict_cuda_benchmark_qualification_review",
    "dense_gguf_qwen_benchmark_qualification_review",
];

fn is_cuda_benchmark_device_label(label: &str) -> bool {
    matches!(label, "cuda" | RTX_5070_TI_CUDA)
}

/// Benchmark command arguments
#[derive(Args, Debug)]
pub struct BenchmarkCommand {
    /// Path to the model file; optional when reporting a governed CUDA benchmark receipt
    #[arg(short, long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    #[arg(short, long, value_name = "DEVICE", help = BENCHMARK_DEVICE_HELP)]
    pub device: Option<String>,

    /// Number of benchmark iterations
    #[arg(long, default_value = "10", value_name = "N")]
    pub iterations: usize,

    /// Warmup iterations
    #[arg(long, default_value = "3", value_name = "N")]
    pub warmup: usize,

    /// Benchmark prompt length
    #[arg(long, default_value = "128", value_name = "TOKENS")]
    pub prompt_length: usize,

    /// Generation length
    #[arg(long, default_value = "256", value_name = "TOKENS")]
    pub generation_length: usize,

    /// Compare against Python baseline
    #[arg(long)]
    pub compare_python: bool,

    /// Generate flamegraph
    #[arg(long)]
    pub flamegraph: bool,

    /// Output format (text, json, csv)
    #[arg(long, default_value = "text", value_name = "FORMAT")]
    pub format: String,

    /// Output file for results
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Governed CUDA benchmark receipt to validate and report instead of running the legacy CPU benchmark
    #[arg(long, value_name = "PATH")]
    pub cuda_benchmark_receipt: Option<PathBuf>,

    /// Profile to select from a governed CUDA benchmark receipt report
    #[arg(long, value_name = "PROFILE")]
    pub profile: Option<String>,

    /// Memory profiling
    #[arg(long)]
    pub memory_profile: bool,

    /// Batch sizes to test
    #[arg(long, value_delimiter = ',', default_values = ["1", "4", "8"])]
    pub batch_sizes: Vec<usize>,

    /// Sequence lengths to test
    #[arg(long, value_delimiter = ',', default_values = ["128", "512", "1024"])]
    pub sequence_lengths: Vec<usize>,
}

/// Benchmark results
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub model_path: String,
    pub device: String,
    pub timestamp: String,
    pub system_info: SystemInfo,
    pub benchmark_config: BenchmarkConfig,
    pub results: Vec<BenchmarkResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub rust_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub iterations: usize,
    pub warmup: usize,
    pub prompt_length: usize,
    pub generation_length: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub iterations: Vec<IterationResult>,
    pub statistics: Statistics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IterationResult {
    pub iteration: usize,
    pub latency_ms: f64,
    pub tokens_per_second: f64,
    pub memory_used_mb: Option<f64>,
    pub peak_memory_mb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Statistics {
    pub mean_latency_ms: f64,
    pub std_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub mean_tokens_per_second: f64,
    pub std_tokens_per_second: f64,
    pub peak_memory_mb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_tests: usize,
    pub total_duration_s: f64,
    pub best_performance: BestPerformance,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BestPerformance {
    pub test_name: String,
    pub tokens_per_second: f64,
    pub latency_ms: f64,
    pub batch_size: usize,
    pub sequence_length: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaBenchmarkReceiptReport {
    pub receipt_path: String,
    pub artifact_kind: String,
    pub claim: String,
    pub selected_backend: String,
    pub selected_route: Option<String>,
    pub runtime_api: String,
    pub fallback_used: bool,
    pub speedup_claim: bool,
    pub benchmark_qualified_speedup: bool,
    pub full_cuda_residency_claimed: bool,
    pub profile_count: usize,
    pub profiles: Vec<CudaBenchmarkProfileReport>,
    pub qualification_status: Option<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaBenchmarkProfileReport {
    pub profile: String,
    pub decision: Option<String>,
    pub cpu_total_ms_mean: Option<f64>,
    pub cuda_total_ms_mean: Option<f64>,
    pub cuda_total_ms: Option<f64>,
    pub cuda_kernel_time_ms: Option<f64>,
    pub host_to_device_bytes: Option<u64>,
    pub device_to_host_bytes: Option<u64>,
    pub host_to_device_ms: Option<f64>,
    pub device_to_host_ms: Option<f64>,
    pub quality_passed: Option<bool>,
    pub fallback_free: Option<bool>,
    pub benchmark_qualified_speedup: Option<bool>,
}

impl BenchmarkCommand {
    /// Execute the benchmark command
    pub async fn execute(&self, config: &CliConfig) -> Result<()> {
        // Validate arguments
        self.validate_args()?;

        if self.cuda_benchmark_receipt.is_some() {
            return self.execute_cuda_benchmark_receipt_report(config).await;
        }

        let device_label = self.requested_device_label(config);
        if is_cuda_benchmark_device_label(device_label) {
            anyhow::bail!(
                "{}",
                unsupported_benchmark_device_message(device_label, self.profile.as_deref())
            );
        }

        let model_path = self.model_path()?;
        info!("Starting benchmark for model: {}", model_path.display());

        // Load model and tokenizer
        let (mut engine, _tokenizer) = self.load_model_and_tokenizer(config).await?;

        // Run benchmarks
        let results = self.run_benchmarks(&mut engine).await?;

        // Generate flamegraph if requested
        if self.flamegraph {
            self.generate_flamegraph().await?;
        }

        // Compare with Python if requested
        if self.compare_python {
            self.compare_with_python(&results).await?;
        }

        // Output results
        self.output_results(&results).await?;

        Ok(())
    }

    /// Validate command arguments
    fn validate_args(&self) -> Result<()> {
        if let Some(model) = &self.model {
            if !model.exists() {
                anyhow::bail!("Model file does not exist: {}", model.display());
            }
        } else if self.cuda_benchmark_receipt.is_none() {
            anyhow::bail!("--model <PATH> is required unless --cuda-benchmark-receipt is provided");
        }

        // Validate format
        match self.format.as_str() {
            "text" | "json" | "csv" => {}
            _ => anyhow::bail!("Invalid format: {}. Must be one of: text, json, csv", self.format),
        }

        // Validate iterations
        if self.iterations == 0 {
            anyhow::bail!("Iterations must be greater than 0");
        }

        // Validate batch sizes
        for &batch_size in &self.batch_sizes {
            if batch_size == 0 {
                anyhow::bail!("Batch size must be greater than 0");
            }
        }

        // Validate sequence lengths
        for &seq_len in &self.sequence_lengths {
            if seq_len == 0 {
                anyhow::bail!("Sequence length must be greater than 0");
            }
        }

        Ok(())
    }

    fn requested_device_label<'a>(&'a self, config: &'a CliConfig) -> &'a str {
        self.device.as_deref().unwrap_or(config.default_device.as_str())
    }

    fn model_path(&self) -> Result<&Path> {
        self.model
            .as_deref()
            .context("--model <PATH> is required unless --cuda-benchmark-receipt is provided")
    }

    async fn execute_cuda_benchmark_receipt_report(&self, config: &CliConfig) -> Result<()> {
        let device_label = self.requested_device_label(config);
        if !is_cuda_benchmark_device_label(device_label) {
            anyhow::bail!(
                "--cuda-benchmark-receipt requires --device cuda or --device {RTX_5070_TI_CUDA}; got {device_label}"
            );
        }

        let receipt_path =
            self.cuda_benchmark_receipt.as_ref().expect("checked cuda_benchmark_receipt presence");
        let receipt = read_benchmark_receipt_json(receipt_path)?;
        let mut report =
            cuda_benchmark_receipt_report(receipt_path, &receipt).with_context(|| {
                format!("invalid governed CUDA benchmark receipt: {}", receipt_path.display())
            })?;
        if let Some(profile) = &self.profile {
            filter_cuda_benchmark_receipt_report_profile(&mut report, profile)?;
        }

        self.output_cuda_benchmark_receipt_report(receipt_path, &receipt, &report).await
    }

    /// Load model and tokenizer
    async fn load_model_and_tokenizer(
        &self,
        config: &CliConfig,
    ) -> Result<(InferenceEngine, std::sync::Arc<dyn bitnet_tokenizers::Tokenizer>)> {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
        pb.set_message("Loading model for benchmarking...");
        pb.enable_steady_tick(Duration::from_millis(100));

        // Determine device
        let device = self.determine_device(config)?;
        debug!("Using device: {:?}", device);

        // Load model
        let loader = ModelLoader::new(bitnet_common::Device::from(&device));
        let model_path = self.model_path()?;
        let model = loader
            .load(model_path)
            .with_context(|| format!("Failed to load model: {}", model_path.display()))?;

        // Load tokenizer
        let tokenizer =
            TokenizerBuilder::from_pretrained("gpt2").context("Failed to load tokenizer")?;

        // Create inference engine
        let model_arc: Arc<dyn bitnet_models::Model> = model.into();
        let tokenizer_arc: Arc<dyn Tokenizer> = tokenizer.clone();
        let bn_device = bitnet_common::Device::from(&device);
        let engine = InferenceEngine::new(model_arc, tokenizer_arc, bn_device)
            .context("Failed to create inference engine")?;

        pb.finish_with_message(format!("{} Model loaded for benchmarking", style("✓").green()));

        Ok((engine, tokenizer))
    }

    /// Determine device to use
    fn determine_device(&self, config: &CliConfig) -> Result<Device> {
        let device_str = self.device.as_ref().unwrap_or(&config.default_device);

        match device_str.as_str() {
            "cpu" | "auto" => {
                info!("Using CPU device for benchmarking");
                Ok(Device::Cpu)
            }
            label if is_supported_device_label(label) => {
                anyhow::bail!(
                    "{}",
                    unsupported_benchmark_device_message(label, self.profile.as_deref())
                )
            }
            _ => anyhow::bail!("{}", invalid_device_message(device_str)),
        }
    }

    /// Run all benchmarks
    async fn run_benchmarks(&self, _engine: &mut InferenceEngine) -> Result<BenchmarkResults> {
        let start_time = Instant::now();
        let mut all_results = Vec::new();

        // Calculate total tests
        let total_tests = self.batch_sizes.len() * self.sequence_lengths.len();

        let pb = ProgressBar::new(total_tests as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        // Run benchmarks for each combination
        for &batch_size in &self.batch_sizes {
            for &seq_len in &self.sequence_lengths {
                let test_name = format!("batch_{}_seq_{}", batch_size, seq_len);
                pb.set_message(format!("Running {}", test_name));

                let result = self.run_single_benchmark(&test_name, batch_size, seq_len).await?;
                all_results.push(result);

                pb.inc(1);
            }
        }

        pb.finish_with_message(format!("{} All benchmarks completed", style("✓").green()));

        // Calculate summary
        let summary = self.calculate_summary(&all_results, start_time.elapsed());

        Ok(BenchmarkResults {
            model_path: self.model_path()?.display().to_string(),
            device: self.device.clone().unwrap_or_else(|| "cpu".to_string()),
            timestamp: chrono::Utc::now().to_rfc3339(),
            system_info: self.get_system_info(),
            benchmark_config: BenchmarkConfig {
                iterations: self.iterations,
                warmup: self.warmup,
                prompt_length: self.prompt_length,
                generation_length: self.generation_length,
            },
            results: all_results,
            summary,
        })
    }

    /// Run a single benchmark configuration
    async fn run_single_benchmark(
        &self,
        test_name: &str,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<BenchmarkResult> {
        let mut iterations = Vec::new();

        // Warmup iterations
        for i in 0..self.warmup {
            debug!("Warmup iteration {} for {}", i + 1, test_name);
            self.run_single_iteration(i, batch_size, seq_len, true).await?;
        }

        // Actual benchmark iterations
        for i in 0..self.iterations {
            let result = self.run_single_iteration(i, batch_size, seq_len, false).await?;
            iterations.push(result);
        }

        // Calculate statistics
        let statistics = self.calculate_statistics(&iterations);

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            batch_size,
            sequence_length: seq_len,
            iterations,
            statistics,
        })
    }

    /// Run a single iteration
    async fn run_single_iteration(
        &self,
        iteration: usize,
        batch_size: usize,
        seq_len: usize,
        is_warmup: bool,
    ) -> Result<IterationResult> {
        let start_time = Instant::now();

        // Simulate inference work
        let work_duration = Duration::from_millis((50 + batch_size * 10 + seq_len / 10) as u64);
        tokio::time::sleep(work_duration).await;

        let elapsed = start_time.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;

        // Calculate tokens per second (simulated)
        let total_tokens = batch_size * seq_len;
        let tokens_per_second = total_tokens as f64 / elapsed.as_secs_f64();

        // Simulate memory usage
        let memory_used_mb = if self.memory_profile {
            Some(100.0 + (batch_size as f64 * seq_len as f64 * 0.01))
        } else {
            None
        };

        let peak_memory_mb = memory_used_mb.map(|m| m * 1.2);

        if !is_warmup {
            debug!(
                "Iteration {}: {:.2}ms, {:.2} tok/s",
                iteration + 1,
                latency_ms,
                tokens_per_second
            );
        }

        Ok(IterationResult {
            iteration,
            latency_ms,
            tokens_per_second,
            memory_used_mb,
            peak_memory_mb,
        })
    }

    /// Calculate statistics from iterations
    fn calculate_statistics(&self, iterations: &[IterationResult]) -> Statistics {
        let latencies: Vec<f64> = iterations.iter().map(|r| r.latency_ms).collect();
        let throughputs: Vec<f64> = iterations.iter().map(|r| r.tokens_per_second).collect();

        let mean_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let mean_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;

        let std_latency = {
            let variance = latencies.iter().map(|&x| (x - mean_latency).powi(2)).sum::<f64>()
                / latencies.len() as f64;
            variance.sqrt()
        };

        let std_throughput = {
            let variance = throughputs.iter().map(|&x| (x - mean_throughput).powi(2)).sum::<f64>()
                / throughputs.len() as f64;
            variance.sqrt()
        };

        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&sorted_latencies, 50.0);
        let p95 = percentile(&sorted_latencies, 95.0);
        let p99 = percentile(&sorted_latencies, 99.0);

        let peak_memory = iterations.iter().filter_map(|r| r.peak_memory_mb).fold(0.0f64, f64::max);

        Statistics {
            mean_latency_ms: mean_latency,
            std_latency_ms: std_latency,
            min_latency_ms: sorted_latencies[0],
            max_latency_ms: sorted_latencies[sorted_latencies.len() - 1],
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            mean_tokens_per_second: mean_throughput,
            std_tokens_per_second: std_throughput,
            peak_memory_mb: if peak_memory > 0.0 { Some(peak_memory) } else { None },
        }
    }

    /// Calculate benchmark summary
    fn calculate_summary(
        &self,
        results: &[BenchmarkResult],
        total_duration: Duration,
    ) -> BenchmarkSummary {
        // Find best performance
        let best = results
            .iter()
            .max_by(|a, b| {
                a.statistics
                    .mean_tokens_per_second
                    .partial_cmp(&b.statistics.mean_tokens_per_second)
                    .unwrap()
            })
            .unwrap();

        let best_performance = BestPerformance {
            test_name: best.test_name.clone(),
            tokens_per_second: best.statistics.mean_tokens_per_second,
            latency_ms: best.statistics.mean_latency_ms,
            batch_size: best.batch_size,
            sequence_length: best.sequence_length,
        };

        // Generate recommendations
        let mut recommendations = Vec::new();

        if best.batch_size > 1 {
            recommendations
                .push(format!("Best performance achieved with batch size {}", best.batch_size));
        }

        if best.statistics.mean_tokens_per_second > 100.0 {
            recommendations.push(
                "Good throughput achieved. Consider GPU acceleration for even better performance."
                    .to_string(),
            );
        } else {
            recommendations.push(
                "Consider optimizing model or using GPU acceleration for better performance."
                    .to_string(),
            );
        }

        if let Some(peak_memory) = best.statistics.peak_memory_mb
            && peak_memory > 1000.0
        {
            recommendations.push(
                "High memory usage detected. Consider using quantization or smaller batch sizes."
                    .to_string(),
            );
        }

        BenchmarkSummary {
            total_tests: results.len(),
            total_duration_s: total_duration.as_secs_f64(),
            best_performance,
            recommendations,
        }
    }

    /// Get system information
    fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_cores: num_cpus::get(),
            memory_gb: 16.0, // Placeholder
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Generate flamegraph
    async fn generate_flamegraph(&self) -> Result<()> {
        info!("Generating flamegraph...");

        // Placeholder implementation
        println!("{} Flamegraph generation not yet implemented", style("⚠").yellow());
        println!("  To generate flamegraphs, use:");
        println!("  cargo install flamegraph");
        println!("  sudo flamegraph -- bitnet benchmark --model {}", self.model_path()?.display());

        Ok(())
    }

    /// Compare with Python baseline
    async fn compare_with_python(&self, _results: &BenchmarkResults) -> Result<()> {
        info!("Comparing with Python baseline...");

        // Placeholder implementation
        println!("{} Python comparison not yet implemented", style("⚠").yellow());
        println!("  To compare with Python:");
        println!("  1. Run the original Python implementation");
        println!("  2. Compare the results manually");

        Ok(())
    }

    /// Output results in the specified format
    async fn output_results(&self, results: &BenchmarkResults) -> Result<()> {
        let output: Box<dyn Write> = if let Some(output_path) = &self.output {
            Box::new(std::fs::File::create(output_path).with_context(|| {
                format!("Failed to create output file: {}", output_path.display())
            })?)
        } else {
            Box::new(std::io::stdout())
        };

        match self.format.as_str() {
            "json" => {
                serde_json::to_writer_pretty(output, results)?;
            }
            "csv" => {
                self.write_csv_results(output, results)?;
            }
            _ => {
                self.write_text_results(output, results)?;
            }
        }

        Ok(())
    }

    async fn output_cuda_benchmark_receipt_report(
        &self,
        receipt_path: &Path,
        receipt: &Value,
        report: &CudaBenchmarkReceiptReport,
    ) -> Result<()> {
        let output: Box<dyn Write> = if let Some(output_path) = &self.output {
            Box::new(std::fs::File::create(output_path).with_context(|| {
                format!("Failed to create output file: {}", output_path.display())
            })?)
        } else {
            Box::new(std::io::stdout())
        };

        match self.format.as_str() {
            "json" => serde_json::to_writer_pretty(output, report)?,
            "csv" => self.write_cuda_benchmark_receipt_csv(output, report)?,
            _ => self.write_cuda_benchmark_receipt_text(output, receipt_path, receipt, report)?,
        }

        Ok(())
    }

    fn write_cuda_benchmark_receipt_text(
        &self,
        mut output: Box<dyn Write>,
        receipt_path: &Path,
        receipt: &Value,
        report: &CudaBenchmarkReceiptReport,
    ) -> Result<()> {
        writeln!(output, "\n{}", style("CUDA Benchmark Receipt Report").bold().cyan())?;
        writeln!(output, "================================")?;
        writeln!(output)?;
        writeln!(output, "Receipt: {}", report.receipt_path)?;
        writeln!(output, "Artifact: {}", report.artifact_kind)?;
        writeln!(output, "Claim: {}", report.claim)?;
        writeln!(output, "Backend: {}", report.selected_backend)?;
        if let Some(route) = &report.selected_route {
            writeln!(output, "Route: {route}")?;
        }
        writeln!(output, "Runtime: {}", report.runtime_api)?;
        writeln!(output, "Fallback: {}", report.fallback_used)?;
        writeln!(output, "Speedup claim: {}", report.speedup_claim)?;
        writeln!(output, "Benchmark-qualified speedup: {}", report.benchmark_qualified_speedup)?;
        writeln!(output, "Full CUDA residency claimed: {}", report.full_cuda_residency_claimed)?;
        if let Some(status) = &report.qualification_status {
            writeln!(output, "Qualification status: {status}")?;
        }
        writeln!(output, "Claim boundary: {}", report.claim_boundary)?;

        if !report.profiles.is_empty() {
            writeln!(output)?;
            writeln!(output, "Profiles:")?;
            for profile in &report.profiles {
                writeln!(output, "  - {}", profile.profile)?;
                if let Some(decision) = &profile.decision {
                    writeln!(output, "    decision: {decision}")?;
                }
                if let Some(cpu_ms) = profile.cpu_total_ms_mean {
                    writeln!(output, "    cpu_mean_total_ms: {cpu_ms:.3}")?;
                }
                if let Some(cuda_ms) = profile.cuda_total_ms_mean.or(profile.cuda_total_ms) {
                    writeln!(output, "    cuda_total_ms: {cuda_ms:.3}")?;
                }
                if let Some(kernel_ms) = profile.cuda_kernel_time_ms {
                    writeln!(output, "    cuda_kernel_time_ms: {kernel_ms:.3}")?;
                }
                if let Some(h2d_ms) = profile.host_to_device_ms {
                    writeln!(output, "    host_to_device_ms: {h2d_ms:.3}")?;
                }
                if let Some(d2h_ms) = profile.device_to_host_ms {
                    writeln!(output, "    device_to_host_ms: {d2h_ms:.3}")?;
                }
                if let Some(h2d_bytes) = profile.host_to_device_bytes {
                    writeln!(output, "    host_to_device_bytes: {h2d_bytes}")?;
                }
                if let Some(d2h_bytes) = profile.device_to_host_bytes {
                    writeln!(output, "    device_to_host_bytes: {d2h_bytes}")?;
                }
                if let Some(quality) = profile.quality_passed {
                    writeln!(output, "    quality_passed: {quality}")?;
                }
                if let Some(qualified) = profile.benchmark_qualified_speedup {
                    writeln!(output, "    benchmark_qualified_speedup: {qualified}")?;
                }
            }
        }

        writeln!(output)?;
        let explanation = explain_receipt(receipt_path, receipt);
        for line in super::receipts::compact_proof_lines(&explanation) {
            writeln!(output, "{line}")?;
        }
        Ok(())
    }

    fn write_cuda_benchmark_receipt_csv(
        &self,
        mut output: Box<dyn Write>,
        report: &CudaBenchmarkReceiptReport,
    ) -> Result<()> {
        writeln!(
            output,
            "profile,decision,cpu_total_ms_mean,cuda_total_ms_mean,cuda_kernel_time_ms,host_to_device_bytes,device_to_host_bytes,host_to_device_ms,device_to_host_ms,quality_passed,benchmark_qualified_speedup"
        )?;

        for profile in &report.profiles {
            writeln!(
                output,
                "{},{},{},{},{},{},{},{},{},{},{}",
                profile.profile,
                profile.decision.as_deref().unwrap_or(""),
                optional_f64(profile.cpu_total_ms_mean),
                optional_f64(profile.cuda_total_ms_mean.or(profile.cuda_total_ms)),
                optional_f64(profile.cuda_kernel_time_ms),
                optional_u64(profile.host_to_device_bytes),
                optional_u64(profile.device_to_host_bytes),
                optional_f64(profile.host_to_device_ms),
                optional_f64(profile.device_to_host_ms),
                optional_bool(profile.quality_passed),
                optional_bool(profile.benchmark_qualified_speedup)
            )?;
        }

        Ok(())
    }

    /// Write results in text format
    fn write_text_results(
        &self,
        mut output: Box<dyn Write>,
        results: &BenchmarkResults,
    ) -> Result<()> {
        writeln!(output, "\n{}", style("BitNet Benchmark Results").bold().cyan())?;
        writeln!(output, "================================")?;
        writeln!(output)?;

        // System info
        writeln!(output, "{}", style("System Information:").bold())?;
        writeln!(output, "  Model: {}", results.model_path)?;
        writeln!(output, "  Device: {}", results.device)?;
        writeln!(output, "  OS: {} ({})", results.system_info.os, results.system_info.arch)?;
        writeln!(output, "  CPU Cores: {}", results.system_info.cpu_cores)?;
        writeln!(output, "  Timestamp: {}", results.timestamp)?;
        writeln!(output)?;

        // Benchmark config
        writeln!(output, "{}", style("Benchmark Configuration:").bold())?;
        writeln!(output, "  Iterations: {}", results.benchmark_config.iterations)?;
        writeln!(output, "  Warmup: {}", results.benchmark_config.warmup)?;
        writeln!(output, "  Prompt Length: {}", results.benchmark_config.prompt_length)?;
        writeln!(output, "  Generation Length: {}", results.benchmark_config.generation_length)?;
        writeln!(output)?;

        // Results
        writeln!(output, "{}", style("Results:").bold())?;
        for result in &results.results {
            writeln!(output, "  {}:", style(&result.test_name).bold())?;
            writeln!(output, "    Mean Latency: {:.2} ms", result.statistics.mean_latency_ms)?;
            writeln!(output, "    Std Latency: {:.2} ms", result.statistics.std_latency_ms)?;
            writeln!(output, "    P95 Latency: {:.2} ms", result.statistics.p95_latency_ms)?;
            writeln!(
                output,
                "    Mean Throughput: {:.2} tokens/sec",
                result.statistics.mean_tokens_per_second
            )?;
            if let Some(memory) = result.statistics.peak_memory_mb {
                writeln!(output, "    Peak Memory: {:.2} MB", memory)?;
            }
            writeln!(output)?;
        }

        // Summary
        writeln!(output, "{}", style("Summary:").bold())?;
        writeln!(output, "  Total Tests: {}", results.summary.total_tests)?;
        writeln!(output, "  Total Duration: {:.2}s", results.summary.total_duration_s)?;
        writeln!(
            output,
            "  Best Performance: {} ({:.2} tokens/sec)",
            results.summary.best_performance.test_name,
            results.summary.best_performance.tokens_per_second
        )?;
        writeln!(output)?;

        // Recommendations
        if !results.summary.recommendations.is_empty() {
            writeln!(output, "{}", style("Recommendations:").bold())?;
            for rec in &results.summary.recommendations {
                writeln!(output, "  • {}", rec)?;
            }
        }

        Ok(())
    }

    /// Write results in CSV format
    fn write_csv_results(
        &self,
        mut output: Box<dyn Write>,
        results: &BenchmarkResults,
    ) -> Result<()> {
        writeln!(
            output,
            "test_name,batch_size,sequence_length,mean_latency_ms,std_latency_ms,p95_latency_ms,mean_tokens_per_second,peak_memory_mb"
        )?;

        for result in &results.results {
            writeln!(
                output,
                "{},{},{},{:.2},{:.2},{:.2},{:.2},{}",
                result.test_name,
                result.batch_size,
                result.sequence_length,
                result.statistics.mean_latency_ms,
                result.statistics.std_latency_ms,
                result.statistics.p95_latency_ms,
                result.statistics.mean_tokens_per_second,
                result
                    .statistics
                    .peak_memory_mb
                    .map(|m| format!("{:.2}", m))
                    .unwrap_or_else(|| "".to_string())
            )?;
        }

        Ok(())
    }
}

fn read_benchmark_receipt_json(path: &Path) -> Result<Value> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse receipt JSON {}", path.display()))
}

fn cuda_benchmark_receipt_report(
    receipt_path: &Path,
    receipt: &Value,
) -> Result<CudaBenchmarkReceiptReport> {
    let artifact_kind = required_str(receipt, &["artifact_kind"])?;
    if !GOVERNED_CUDA_BENCHMARK_ARTIFACTS.contains(&artifact_kind) {
        anyhow::bail!(
            "artifact_kind={artifact_kind} is not a governed CUDA benchmark receipt accepted by `bitnet bench --device cuda --cuda-benchmark-receipt`"
        );
    }

    let claim = required_str(receipt, &["claim"])?;
    let selected_backend = required_str(receipt, &["selected_backend"])
        .or_else(|_| required_str(receipt, &["execution_plan", "selected_backend"]))?;
    if selected_backend != RTX_5070_TI_CUDA {
        anyhow::bail!("selected_backend must be {RTX_5070_TI_CUDA}, got {selected_backend}");
    }

    let runtime_api = required_str(receipt, &["runtime_api"])
        .or_else(|_| required_str(receipt, &["execution_plan", "runtime_api"]))?;
    if runtime_api != "cuda" {
        anyhow::bail!("runtime_api must be cuda, got {runtime_api}");
    }

    let fallback_used = required_bool(receipt, &["fallback_used"])
        .or_else(|_| required_bool(receipt, &["execution_plan", "fallback_used"]))?;
    if fallback_used {
        anyhow::bail!("fallback_used must be false for governed CUDA benchmark reporting");
    }

    let speedup_claim = bool_at_path(receipt, &["speedup_claim"])
        .or_else(|| bool_at_path(receipt, &["claim_boundary", "speedup_claim"]))
        .unwrap_or(false);
    let benchmark_qualified_speedup =
        bool_at_path(receipt, &["benchmark_qualified_speedup"]).unwrap_or(false);
    if speedup_claim && !benchmark_qualified_speedup {
        anyhow::bail!("speedup_claim=true is not accepted unless benchmark_qualified_speedup=true");
    }

    let full_cuda_residency_claimed = bool_at_path(receipt, &["full_cuda_residency_claimed"])
        .or_else(|| bool_at_path(receipt, &["claim_boundary", "full_cuda_residency_claimed"]))
        .or_else(|| bool_at_path(receipt, &["execution_plan", "full_cuda_residency_claimed"]))
        .unwrap_or(false);

    let selected_route = str_at_path(receipt, &["selected_route"])
        .or_else(|| str_at_path(receipt, &["execution_plan", "selected_route"]))
        .map(str::to_string);

    let profiles = cuda_benchmark_profile_reports(receipt);
    if profiles.is_empty() {
        anyhow::bail!("governed CUDA benchmark receipt must include profile evidence");
    }

    let qualification_status = str_at_path(receipt, &["qualification_decision", "status"])
        .or_else(|| str_at_path(receipt, &["benchmark_summary", "status"]))
        .or_else(|| str_at_path(receipt, &["comparator_summary", "status"]))
        .map(str::to_string);

    Ok(CudaBenchmarkReceiptReport {
        receipt_path: receipt_path.display().to_string(),
        artifact_kind: artifact_kind.to_string(),
        claim: claim.to_string(),
        selected_backend: selected_backend.to_string(),
        selected_route,
        runtime_api: runtime_api.to_string(),
        fallback_used,
        speedup_claim,
        benchmark_qualified_speedup,
        full_cuda_residency_claimed,
        profile_count: profiles.len(),
        profiles,
        qualification_status,
        claim_boundary:
            "receipt-backed CUDA benchmark report only; no fresh benchmark execution or new speedup claim"
                .to_string(),
    })
}

fn cuda_benchmark_profile_reports(receipt: &Value) -> Vec<CudaBenchmarkProfileReport> {
    let mut profiles = Vec::new();
    collect_cuda_benchmark_profiles(receipt.get("profile_reviews"), &mut profiles);
    collect_cuda_benchmark_profiles(receipt.get("profiles"), &mut profiles);
    collect_cuda_benchmark_profiles(receipt.get("benchmark_profiles"), &mut profiles);
    profiles
}

fn filter_cuda_benchmark_receipt_report_profile(
    report: &mut CudaBenchmarkReceiptReport,
    profile: &str,
) -> Result<()> {
    if profile.trim().is_empty() {
        anyhow::bail!("--profile must not be empty");
    }

    let available: Vec<String> =
        report.profiles.iter().map(|entry| entry.profile.clone()).collect();
    report.profiles.retain(|entry| entry.profile == profile);
    if report.profiles.is_empty() {
        anyhow::bail!(
            "profile `{profile}` was not found in governed CUDA benchmark receipt; available profiles: {}",
            available.join(", ")
        );
    }
    report.profile_count = report.profiles.len();
    Ok(())
}

fn collect_cuda_benchmark_profiles(
    value: Option<&Value>,
    profiles: &mut Vec<CudaBenchmarkProfileReport>,
) {
    let Some(entries) = value.and_then(Value::as_array) else {
        return;
    };

    for entry in entries {
        let Some(profile) = str_at_path(entry, &["profile"]) else {
            continue;
        };

        profiles.push(CudaBenchmarkProfileReport {
            profile: profile.to_string(),
            decision: str_at_path(entry, &["decision"])
                .or_else(|| str_at_path(entry, &["status"]))
                .map(str::to_string),
            cpu_total_ms_mean: f64_at_path(entry, &["cpu_total_ms_mean"])
                .or_else(|| f64_at_path(entry, &["cpu_total_ms", "mean"])),
            cuda_total_ms_mean: f64_at_path(entry, &["cuda_total_ms_mean"])
                .or_else(|| f64_at_path(entry, &["cuda_total_ms", "mean"])),
            cuda_total_ms: f64_at_path(entry, &["cuda_total_ms"]),
            cuda_kernel_time_ms: f64_at_path(entry, &["cuda_kernel_time_ms"])
                .or_else(|| f64_at_path(entry, &["kernel_time_ms", "mean"])),
            host_to_device_bytes: u64_at_path(entry, &["host_to_device_bytes"])
                .or_else(|| u64_at_path(entry, &["host_to_device_bytes", "mean"])),
            device_to_host_bytes: u64_at_path(entry, &["device_to_host_bytes"])
                .or_else(|| u64_at_path(entry, &["device_to_host_bytes", "mean"])),
            host_to_device_ms: f64_at_path(entry, &["host_to_device_ms"]),
            device_to_host_ms: f64_at_path(entry, &["device_to_host_ms"]),
            quality_passed: bool_at_path(entry, &["quality_passed"]),
            fallback_free: bool_at_path(entry, &["fallback_free"]),
            benchmark_qualified_speedup: bool_at_path(entry, &["benchmark_qualified_speedup"]),
        });
    }
}

fn required_str<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str> {
    str_at_path(value, path).with_context(|| format!("missing string field {}", path.join(".")))
}

fn required_bool(value: &Value, path: &[&str]) -> Result<bool> {
    bool_at_path(value, path).with_context(|| format!("missing bool field {}", path.join(".")))
}

fn str_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    value_at_path(value, path).and_then(Value::as_str)
}

fn bool_at_path(value: &Value, path: &[&str]) -> Option<bool> {
    value_at_path(value, path).and_then(Value::as_bool)
}

fn f64_at_path(value: &Value, path: &[&str]) -> Option<f64> {
    value_at_path(value, path).and_then(Value::as_f64)
}

fn u64_at_path(value: &Value, path: &[&str]) -> Option<u64> {
    value_at_path(value, path).and_then(Value::as_u64)
}

fn value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn optional_f64(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}

fn optional_u64(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn optional_bool(value: Option<bool>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    let index = (p / 100.0) * (sorted_data.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

fn unsupported_benchmark_device_message(device: &str, profile: Option<&str>) -> String {
    let profile_note = profile
        .map(|profile| {
            format!(
                " Profile `{profile}` is recognized only for governed CUDA benchmark receipt reporting until live PERF-005 profile execution lands."
            )
        })
        .unwrap_or_default();
    format!(
        "bitnet benchmark does not support device label '{device}'. This legacy benchmark simulates benchmark work and must not silently fall back to CPU for accelerator requests. Use receipt-backed CUDA paths such as `bitnet ask --device cuda ...`, `bitnet chat --device cuda ...`, or governed CUDA benchmark receipts; CPU fallback cannot count as CUDA execution.{profile_note}"
    )
}
